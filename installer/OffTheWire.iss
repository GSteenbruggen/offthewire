; Inno Setup script for the offline coding agent.
;
;     scripts\build_installer.ps1
;
; Produces build\OffTheWire-Setup-<version>.exe from the PyInstaller output
; in build\dist\OffTheWire.
;
; PER-USER, NOT PER-MACHINE. PrivilegesRequired=lowest installs into
; %LOCALAPPDATA%\Programs and edits only HKCU, so there is no UAC prompt and no
; administrator. For a developer tool that is the better trade twice over: the
; person installing it is the only person who will use it, and an unsigned
; installer asking for elevation is exactly the shape SmartScreen and every
; corporate policy is built to distrust.
;
; What this deliberately does NOT do is install Ollama or pull a model. Ollama
; is a separate product with its own installer and update channel, and the
; models are tens of gigabytes -- the one on this machine is 34 GB, against a
; 75 MB application. Bundling either would produce a download nobody would
; finish for software that would still need the real thing. Instead the last
; page checks for Ollama and says plainly what is missing.

#define AppName        "OffTheWire"
#define AppVersion     "1.4.2"
#define AppPublisher   "Gerri"
#define AppExe         "OffTheWire.exe"
; Overridable with ISCC /DSourceDir=<absolute path>, which build_installer.ps1
; does. The relative default works but reaches the same files via
; "installer\..\build\...", and those 13 extra characters are spent on every
; bundled path -- which matters because the deepest file PyInstaller emits
; (lxml's isoschematron resources) is already within a few characters of
; Windows' 260-character MAX_PATH.
#ifndef SourceDir
  #define SourceDir    "..\build\dist\OffTheWire"
#endif

[Setup]
; Never change AppId: it is how Windows recognises an existing install and
; upgrades it in place rather than leaving two copies behind.
AppId={{1CC2315A-7952-4C30-805C-6D2604D885DF}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
; Shown in the exe's file properties and in Add/Remove Programs.
AppCopyright=Copyright (C) 2026 {#AppPublisher} - MIT License
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; The app is a terminal command; a directory page is still worth keeping so a
; portable install onto another drive stays possible.
DisableDirPage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\build
OutputBaseFilename={#AppName}-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Tells Windows to broadcast the environment change, so a newly opened terminal
; picks up PATH without a sign-out.
ChangesEnvironment=yes
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add to PATH, so `{#AppName}` works in any terminal"; GroupDescription: "Command line:"
Name: "startmenu"; Description: "Create a Start Menu shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked
; Unchecked by default, and deliberately so: it needs Docker Desktop installed
; and running, and it pulls a ~1 GB image. The agent is fully functional
; without it -- only --web is affected.
Name: "searxng"; Description: "Set up web lookup (needs Docker Desktop, downloads ~1 GB)"; GroupDescription: "Optional:"; Flags: unchecked

[Files]
; recursesubdirs picks up PyInstaller's _internal folder, which is most of the
; application; without it the exe installs and immediately fails to start.
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Shipped whether or not the task is ticked, so web lookup can still be set up
; later without needing the project source.
Source: "..\scripts\setup_searxng.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Run]
; Config goes to the writable data directory, not {app}: SearXNG needs to write
; to the volume it mounts, and the install directory is read-only territory.
Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup_searxng.ps1"" -ConfigDir ""{localappdata}\{#AppName}\searxng-config"" -Pause"; \
    StatusMsg: "Setting up web lookup (pulling the SearXNG image)..."; \
    Flags: waituntilterminated skipifsilent; \
    Tasks: searxng

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{userdocs}"; Tasks: startmenu

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
    ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
const
  EnvironmentKey = 'Environment';

// True when {app} is not already on the user PATH. Without this, reinstalling
// appends a second identical entry every time. Note the // comments throughout
// this section: Inno treats "{" as a comment opener, so a brace comment
// containing "{app}" is terminated early by that constant's own closing brace.
function NeedsAddPath(Dir: string): Boolean;
var
  Existing: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Existing) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(RemoveBackslash(Dir)) + ';',
                ';' + Uppercase(Existing) + ';') = 0;
end;

// Rebuild PATH without our entry, comparing entry by entry.
//
// The usual trick of deleting a matched substring gets the delimiters wrong at
// the head and tail of the string and can leave a stray semicolon or eat a
// neighbouring entry. Splitting and rejoining cannot.
procedure RemoveFromPath(const Dir: string);
var
  Remaining, Rebuilt, Part: string;
  Cut: Integer;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Remaining) then
    exit;
  Rebuilt := '';
  Remaining := Remaining + ';';
  repeat
    Cut := Pos(';', Remaining);
    Part := Trim(Copy(Remaining, 1, Cut - 1));
    Delete(Remaining, 1, Cut);
    if (Part <> '') and
       (CompareText(RemoveBackslash(Part), RemoveBackslash(Dir)) <> 0) then
    begin
      if Rebuilt <> '' then
        Rebuilt := Rebuilt + ';';
      Rebuilt := Rebuilt + Part;
    end;
  until Remaining = '';
  RegWriteExpandStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Rebuilt);
end;

// Is Ollama on this machine? Checked two ways because they fail differently:
// "where" misses an Ollama that installed without updating this process's PATH,
// and the fixed path misses one installed somewhere unusual.
function OllamaInstalled(): Boolean;
var
  Code: Integer;
begin
  Result := FileExists(ExpandConstant('{localappdata}\Programs\Ollama\ollama.exe'))
         or FileExists(ExpandConstant('{pf}\Ollama\ollama.exe'));
  if Result then
    exit;
  Result := Exec(ExpandConstant('{cmd}'), '/c where ollama', '',
                 SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
end;

function DockerPresent(): Boolean;
var
  Code: Integer;
begin
  Result := Exec(ExpandConstant('{cmd}'), '/c where docker', '',
                 SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
end;

// Catch the missing prerequisite on the Ready page rather than after the files
// are copied. Ticking the box, waiting through an install, and only then being
// told Docker is absent is a worse way to learn it.
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID <> wpReady then
    exit;
  if not WizardIsTaskSelected('searxng') then
    exit;
  if DockerPresent() then
    exit;
  if MsgBox('Web lookup needs Docker Desktop, which was not found.' + #13#10 + #13#10 +
            'You can install {#AppName} now and set up web lookup later by running'
            + #13#10 + 'setup_searxng.ps1 from the install folder.' + #13#10 + #13#10 +
            'Continue and try anyway?', mbConfirmation, MB_YESNO) = IDNO then
    Result := False;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    exit;
  if OllamaInstalled() then
    exit;
  // Stated as the remaining step rather than an error: the install itself
  // succeeded, and this is the one thing left before the agent can run.
  MsgBox('{#AppName} is installed, but Ollama was not found.' + #13#10 + #13#10 +
         'The agent runs models locally through Ollama, so it needs:' + #13#10 +
         '  1. Ollama, from https://ollama.com/download' + #13#10 +
         '  2. A model that supports tools, e.g.:' + #13#10 +
         '       ollama pull qwen3.8:27b' + #13#10 + #13#10 +
         'For image attachments the model must also support vision.' + #13#10 +
         'Run "{#AppName}" afterwards and it will list what it finds.',
         mbInformation, MB_OK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: string;
  ResultCode: Integer;
begin
  if CurUninstallStep <> usPostUninstall then
    exit;

  RemoveFromPath(ExpandConstant('{app}'));

  // The SearXNG container is created with --restart unless-stopped, so left
  // alone it keeps running and comes back on every reboot -- long after the
  // program that wanted it is gone. Uninstalling has to offer to take it with
  // us, or this quietly becomes a permanent background service nobody
  // remembers installing.
  if (not UninstallSilent) and DockerPresent() then
  begin
    if Exec(ExpandConstant('{cmd}'),
            '/c docker ps -a --filter "name=searxng-agent" --format "{{.Names}}" | findstr searxng-agent',
            '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0) then
    begin
      if MsgBox('Also remove the SearXNG web-lookup container?' + #13#10 + #13#10 +
                'It was created for {#AppName} and restarts with Docker, so it '
                + 'will keep running otherwise.', mbConfirmation, MB_YESNO) = IDYES then
        Exec(ExpandConstant('{cmd}'), '/c docker rm -f searxng-agent', '',
             SW_HIDE, ewWaitUntilTerminated, ResultCode);
    end;
  end;

  // Saved conversations are the user's work, not ours, and they live outside
  // the install directory precisely so an uninstall cannot take them by
  // accident. Ask.
  DataDir := ExpandConstant('{localappdata}\{#AppName}');
  if not DirExists(DataDir) then
    exit;

  // A silent uninstall suppresses message boxes and takes each default, and
  // the default for this one is Yes -- so an unattended removal would delete
  // every saved conversation without anyone seeing the question. When nobody
  // is there to answer, keep the data: an uninstaller that leaves a directory
  // behind is a nuisance, one that destroys work is not recoverable.
  if UninstallSilent then
    exit;
  if MsgBox('Also delete your saved conversations and pasted images?' + #13#10 + #13#10 +
            DataDir + #13#10 + #13#10 +
            'Choose No to keep them for a future install.',
            mbConfirmation, MB_YESNO) = IDYES then
    DelTree(DataDir, True, True, True);
end;
