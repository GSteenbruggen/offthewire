# Homebrew formula for OffTheWire.
#
# Home: a tap repository named homebrew-offthewire under the GSteenbruggen
# account, containing this file at Formula/offthewire.rb. Users then run:
#
#   brew tap gsteenbruggen/offthewire
#   brew install offthewire
#
# Installing through brew also sidesteps Gatekeeper's quarantine, which is
# the main friction of the raw tarball. The sha256 is the actual hash of the
# published v1.5.0 asset.
class Offthewire < Formula
  desc "Offline coding agent for local Ollama models"
  homepage "https://github.com/GSteenbruggen/offthewire"
  url "https://github.com/GSteenbruggen/offthewire/releases/download/v1.5.0/OffTheWire-1.5.0-macos-arm64.tar.gz"
  sha256 "d20bf976fbc85e64f3fe8ae304ef04741267fd79a248640c224ec2c57d1f301f"
  version "1.5.0"
  license "MIT"

  depends_on arch: :arm64
  depends_on macos: :ventura

  def install
    libexec.install Dir["OffTheWire/*"]
    bin.write_exec_script libexec/"OffTheWire"
    bin.install_symlink bin/"OffTheWire" => "offthewire"
  end

  def caveats
    <<~EOS
      OffTheWire drives models through Ollama, which is installed separately:
        brew install ollama
        ollama pull qwen3.8:27b   (or any model with the `tools` capability)
    EOS
  end

  test do
    assert_match "OffTheWire", shell_output("#{bin}/OffTheWire --version")
  end
end
