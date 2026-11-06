#define MyAppName "SegMed"
#define MyAppVersion "1.5.0"

[Setup]
AppId={{6FBD9B18-94D4-40AA-95CA-1162E79418F9}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=SegMed Project
DefaultDirName={localappdata}\Programs\SegMed
DefaultGroupName=SegMed
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
SetupIconFile=..\resource\imgs\icon.ico
UninstallDisplayIcon={app}\resource\imgs\icon.ico
OutputDir=output
OutputBaseFilename=SegMed-Setup-{#MyAppVersion}-preview
DisableReadyMemo=no
CloseApplications=yes

[Tasks]
Name: "gpu"; Description: "Automatically configure supported NVIDIA GPU acceleration (CUDA 12.8 or 12.6; CPU fallback)"; GroupDescription: "Inference configuration:"
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\launcher.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\uv.lock"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\.python-version"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\modules\*.py"; DestDir: "{app}\modules"; Flags: ignoreversion
Source: "..\resource\imgs\*"; DestDir: "{app}\resource\imgs"; Flags: ignoreversion
Source: "bin\uv.exe"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "install_dependencies.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "verify_environment.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "verify_torch_backend.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "select_torch_backend.py"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\SegMed\SegMed"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\resource\imgs\icon.ico"
Name: "{autoprograms}\SegMed\Repair inference environment"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install_dependencies.ps1"" -ProjectPath ""{app}"" -Backend Preserve"; WorkingDir: "{app}"; IconFilename: "{app}\resource\imgs\icon.ico"
Name: "{autoprograms}\SegMed\Auto-configure NVIDIA acceleration"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install_dependencies.ps1"" -ProjectPath ""{app}"" -Backend Auto"; WorkingDir: "{app}"; IconFilename: "{app}\resource\imgs\icon.ico"
Name: "{autodesktop}\SegMed"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\resource\imgs\icon.ico"; Tasks: desktopicon

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\runtime\python"
Type: files; Name: "{app}\.installation-complete"
Type: files; Name: "{app}\.inference-backend"
Type: files; Name: "{app}\.cuda-device-uuid"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  PowerShell: String;
  Params: String;
  Backend: String;
begin
  if CurStep = ssPostInstall then
  begin
    if WizardIsTaskSelected('gpu') then
      Backend := 'Auto'
    else
      Backend := 'CPU';
    WizardForm.StatusLabel.Caption := 'Installing Python and SegMed dependencies...';
    PowerShell := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' +
      ExpandConstant('{app}\install_dependencies.ps1') +
      '" -ProjectPath "' + ExpandConstant('{app}') + '" -Backend ' + Backend;
    if not Exec(PowerShell, Params, ExpandConstant('{app}'), SW_SHOW,
      ewWaitUntilTerminated, ResultCode) then
      RaiseException('Unable to start dependency installation.');
    if ResultCode <> 0 then
      RaiseException('SegMed dependency setup failed. See ' +
        ExpandConstant('{localappdata}\SegMed\logs\setup.log') +
        '. You can retry using the Repair inference environment shortcut.');
  end;
end;
