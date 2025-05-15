# VSCode Configuration for Running with `uv run`

This directory contains VSCode configuration files that enable running Python files with the `uv run` command. The configuration is designed to work with `uv pin` for switching Python versions.

## Configuration Files

- `settings.json`: Configures VSCode to use `uv run` for running Python files
- `launch.json`: Configures the VSCode debugger to use `uv run` when debugging Python files
- `tasks.json`: Defines tasks for running Python files with `uv run`

## How to Run Python Files with `uv run`

### Method 1: Using the Run Button

1. Open a Python file in the editor
2. Click the "Run" button in the top-right corner of the editor
3. Select "Python: Current File with uv" from the dropdown menu

### Method 2: Using the Command Palette

1. Open a Python file in the editor
2. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac) to open the Command Palette
3. Type "Run Task" and select "Tasks: Run Task"
4. Select one of the following tasks:
   - `uv-run-current-file`: Run the currently open file
   - `uv-run-localfile-asr-text`: Run the `localfile-asr-text.py` file
   - `uv-run-asr-text`: Run the `asr-text.py` file
   - `uv-run-asr-summary`: Run the `asr-summary.py` file

### Method 3: Using Keyboard Shortcuts

1. Open a Python file in the editor
2. Press `F5` to run the file with the debugger
3. Select "Python: Current File with uv" from the dropdown menu

## Debugging with `uv run`

1. Open a Python file in the editor
2. Set breakpoints by clicking in the gutter next to the line numbers
3. Press `F5` to start debugging
4. Select "Python: Debug Current File with uv" from the dropdown menu

## Adding More Tasks

To add more tasks for running specific Python files, edit the `.vscode/tasks.json` file and add a new task with the following format:

```json
{
    "label": "uv-run-your-file-name",
    "type": "shell",
    "command": "uv run --python $(cat .python-version) your-file-name.py",
    "group": "test",
    "presentation": {
        "reveal": "always",
        "panel": "new"
    },
    "problemMatcher": []
}
```

Replace `your-file-name` with the name of your Python file.

## Using with `uv python pin`

This configuration is designed to work with `uv python pin` for switching Python versions. When you use `uv python pin` to switch Python versions, the configuration will automatically use the new Python version.

### How to Switch Python Versions

1. Use the `uv python pin` command to switch Python versions:

   ```bash
   uv python pin 3.11.5  # Replace with the desired Python version
   ```

2. The command will update the `.python-version` file with the new Python version.

3. The VSCode configuration will automatically use the new Python version when running Python files with `uv run`.

### How It Works

- The tasks in `tasks.json` use the `--python $(cat .python-version)` flag to ensure that `uv run` uses the Python version specified in the `.python-version` file.
- The launch configurations in `launch.json` use the `${command:python.interpreterPath}` variable to ensure that the debugger uses the correct Python interpreter.
- The settings in `settings.json` include `"python.terminal.activateEnvInCurrentTerminal": true` to ensure that the terminal uses the correct Python environment.
