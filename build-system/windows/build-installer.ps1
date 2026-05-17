# Q-NarwhalKnight Miner Windows Installer Builder
# Creates MSI installer package for Windows distribution

param(
    [string]$Version = "1.0.0",
    [string]$Configuration = "Release",
    [string]$Platform = "x64"
)

Write-Host "🏗️ Building Q-NarwhalKnight Miner Windows Installer" -ForegroundColor Green
Write-Host "Version: $Version" -ForegroundColor Cyan
Write-Host "Configuration: $Configuration" -ForegroundColor Cyan
Write-Host "Platform: $Platform" -ForegroundColor Cyan

# Set paths
$RootDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$BinDir = "$RootDir\target\x86_64-pc-windows-msvc\release"
$DistDir = "$RootDir\dist"
$TempDir = "$env:TEMP\q-miner-installer"

# Create directories
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

Write-Host "📦 Preparing installer files..." -ForegroundColor Yellow

# Copy binaries
Copy-Item "$BinDir\q-miner.exe" -Destination "$TempDir\"
Copy-Item "$BinDir\q-miner-gui.exe" -Destination "$TempDir\"
Copy-Item "$BinDir\q-miner-benchmark.exe" -Destination "$TempDir\"

# Copy documentation
Copy-Item "$RootDir\crates\q-miner\README.md" -Destination "$TempDir\"
Copy-Item "$RootDir\LICENSE" -Destination "$TempDir\"

# Copy CUDA runtime (if available)
$CudaRuntime = "${env:CUDA_PATH}\bin\cudart64_12.dll"
if (Test-Path $CudaRuntime) {
    Write-Host "📥 Including CUDA runtime" -ForegroundColor Green
    Copy-Item $CudaRuntime -Destination "$TempDir\"
}

# Create WiX installer source
$WixSource = @"
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Id="*" 
           Name="Q-NarwhalKnight Miner" 
           Language="1033" 
           Version="$Version" 
           Manufacturer="Q-NarwhalKnight Labs"
           UpgradeCode="12345678-1234-1234-1234-123456789012">
    
    <Package InstallerVersion="200" 
             Compressed="yes" 
             InstallScope="perMachine"
             Description="High-Performance Quantum-Enhanced Cryptocurrency Miner"
             Comments="CPU/GPU mining with CUDA support for Q-NarwhalKnight network" />

    <MajorUpgrade DowngradeErrorMessage="A newer version is already installed." />
    <MediaTemplate EmbedCab="yes" />

    <Feature Id="ProductFeature" Title="Q-NarwhalKnight Miner" Level="1">
      <ComponentGroupRef Id="ProductComponents" />
      <ComponentRef Id="EnvironmentPath" />
    </Feature>
    
    <Icon Id="QMinerIcon" SourceFile="$RootDir\assets\q-miner.ico" />
    <Property Id="ARPPRODUCTICON" Value="QMinerIcon" />
    <Property Id="ARPHELPLINK" Value="https://github.com/deme-plata/q-narwhalknight" />
    
    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="ProgramFiles64Folder">
        <Directory Id="INSTALLFOLDER" Name="Q-NarwhalKnight Miner" />
      </Directory>
      <Directory Id="ProgramMenuFolder">
        <Directory Id="ApplicationProgramsFolder" Name="Q-NarwhalKnight Miner" />
      </Directory>
      <Directory Id="DesktopFolder" Name="Desktop" />
    </Directory>

    <DirectoryRef Id="INSTALLFOLDER">
      <Component Id="MainExecutable" Guid="*">
        <File Id="QMinerExe" Source="$TempDir\q-miner.exe" KeyPath="yes" />
      </Component>
      
      <Component Id="GuiExecutable" Guid="*">
        <File Id="QMinerGuiExe" Source="$TempDir\q-miner-gui.exe" />
      </Component>
      
      <Component Id="BenchmarkExecutable" Guid="*">
        <File Id="QMinerBenchmarkExe" Source="$TempDir\q-miner-benchmark.exe" />
      </Component>
      
      <Component Id="Documentation" Guid="*">
        <File Id="ReadmeFile" Source="$TempDir\README.md" />
        <File Id="LicenseFile" Source="$TempDir\LICENSE" />
      </Component>
      
      <Component Id="CudaRuntime" Guid="*">
        <File Id="CudaRuntimeDll" Source="$TempDir\cudart64_12.dll" />
      </Component>
    </DirectoryRef>

    <DirectoryRef Id="ApplicationProgramsFolder">
      <Component Id="ApplicationShortcut" Guid="*">
        <Shortcut Id="ApplicationStartMenuShortcut"
                  Name="Q-NarwhalKnight Miner"
                  Description="High-Performance Cryptocurrency Miner"
                  Target="[#QMinerGuiExe]"
                  WorkingDirectory="INSTALLFOLDER" />
        <RemoveFolder Id="ApplicationProgramsFolder" On="uninstall" />
        <RegistryValue Root="HKCU" 
                       Key="Software\Q-NarwhalKnight\Miner" 
                       Name="installed" 
                       Type="integer" 
                       Value="1" 
                       KeyPath="yes" />
      </Component>
    </DirectoryRef>

    <DirectoryRef Id="DesktopFolder">
      <Component Id="DesktopShortcut" Guid="*">
        <Shortcut Id="ApplicationDesktopShortcut"
                  Name="Q-NarwhalKnight Miner"
                  Description="High-Performance Cryptocurrency Miner"
                  Target="[#QMinerGuiExe]"
                  WorkingDirectory="INSTALLFOLDER" />
        <RegistryValue Root="HKCU" 
                       Key="Software\Q-NarwhalKnight\Miner" 
                       Name="desktop_shortcut" 
                       Type="integer" 
                       Value="1" 
                       KeyPath="yes" />
      </Component>
    </DirectoryRef>

    <Component Id="EnvironmentPath" Guid="*" Directory="INSTALLFOLDER">
      <Environment Id="PATH" Name="PATH" Value="[INSTALLFOLDER]" Permanent="no" Part="last" Action="set" System="yes" />
      <RegistryValue Root="HKLM" 
                     Key="Software\Q-NarwhalKnight\Miner" 
                     Name="InstallPath" 
                     Type="string" 
                     Value="[INSTALLFOLDER]" 
                     KeyPath="yes" />
    </Component>

    <ComponentGroup Id="ProductComponents">
      <ComponentRef Id="MainExecutable" />
      <ComponentRef Id="GuiExecutable" />
      <ComponentRef Id="BenchmarkExecutable" />
      <ComponentRef Id="Documentation" />
      <ComponentRef Id="CudaRuntime" />
    </ComponentGroup>
  </Product>
</Wix>
"@

# Save WiX source
$WixFile = "$TempDir\QMinerInstaller.wxs"
$WixSource | Out-File -FilePath $WixFile -Encoding UTF8

Write-Host "🔨 Compiling installer..." -ForegroundColor Yellow

# Check if WiX is installed
if (-not (Get-Command "candle.exe" -ErrorAction SilentlyContinue)) {
    Write-Host "❌ WiX Toolset not found. Installing..." -ForegroundColor Red
    
    # Download and install WiX
    $WixUrl = "https://github.com/wixtoolset/wix3/releases/download/wix3112rtm/wix311-binaries.zip"
    $WixZip = "$env:TEMP\wix311-binaries.zip"
    $WixDir = "$env:TEMP\wix311"
    
    Invoke-WebRequest -Uri $WixUrl -OutFile $WixZip
    Expand-Archive -Path $WixZip -DestinationPath $WixDir -Force
    
    $env:PATH += ";$WixDir"
}

# Compile WiX source
& candle.exe -out "$TempDir\QMinerInstaller.wixobj" "$WixFile"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ WiX compilation failed" -ForegroundColor Red
    exit 1
}

# Link installer
& light.exe -out "$DistDir\Q-NarwhalKnight-Miner-$Version-$Platform.msi" "$TempDir\QMinerInstaller.wixobj"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ WiX linking failed" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Windows installer created successfully!" -ForegroundColor Green
Write-Host "📦 Installer: $DistDir\Q-NarwhalKnight-Miner-$Version-$Platform.msi" -ForegroundColor Cyan

# Create ZIP archive for portable version
Write-Host "📦 Creating portable ZIP archive..." -ForegroundColor Yellow
$ZipPath = "$DistDir\Q-NarwhalKnight-Miner-$Version-$Platform-Portable.zip"
Compress-Archive -Path "$TempDir\*" -DestinationPath $ZipPath -Force

Write-Host "✅ Portable archive created: $ZipPath" -ForegroundColor Green

# Cleanup
Remove-Item -Recurse -Force $TempDir

Write-Host "🎉 Build complete!" -ForegroundColor Green
Write-Host "📋 Build artifacts:" -ForegroundColor Cyan
Write-Host "   • MSI Installer: Q-NarwhalKnight-Miner-$Version-$Platform.msi" -ForegroundColor White
Write-Host "   • Portable ZIP: Q-NarwhalKnight-Miner-$Version-$Platform-Portable.zip" -ForegroundColor White