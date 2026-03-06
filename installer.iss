; Inno Setup Script for ProteinProcessIO
; Download Inno Setup from: https://jrsoftware.org/isinfo.php

#define MyAppName "ProteinProcessIO"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "McGill University"
#define MyAppURL "https://www.eakwofie.com/"
#define MyAppExeName "ProteinProcessIO.exe"

[Setup]
; Application info
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Installation directories
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Output settings
OutputDir=dist\installer
OutputBaseFilename=ProteinProcessIO-{#MyAppVersion}-Setup
; SetupIconFile=src\airclassifier\gui\resources\icon.ico  ; Uncomment when icon.ico exists
Compression=lzma2/ultra64
SolidCompression=yes

; Windows version requirements
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Privileges
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; UI options
WizardStyle=modern

; License and info (optional - uncomment if you have these files)
; LicenseFile=LICENSE.txt
; InfoBeforeFile=README.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Copy all files from the PyInstaller dist folder
Source: "dist\ProteinProcessIO\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu shortcut
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
; Desktop shortcut (optional)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Option to run after installation
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up any generated files on uninstall
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\cache"
Type: filesandordirs; Name: "{app}\__pycache__"

[Code]
// Check if NVIDIA GPU is available (optional warning)
function InitializeSetup(): Boolean;
begin
  Result := True;
  // You could add GPU detection here if needed
end;
