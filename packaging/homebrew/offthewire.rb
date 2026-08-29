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
# published v1.4.2 asset.
class Offthewire < Formula
  desc "Offline coding agent for local Ollama models"
  homepage "https://github.com/GSteenbruggen/offthewire"
  url "https://github.com/GSteenbruggen/offthewire/releases/download/v1.4.2/OffTheWire-1.4.2-macos-arm64.tar.gz"
  sha256 "da7fdb923dc30cccab72e99f4c01b02ab7926e88faba0bad046002907507015c"
  version "1.4.2"
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
