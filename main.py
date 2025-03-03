import os
import subprocess
import json
import shutil
import uuid
import resource
import signal
import boto3
import dotenv
from datetime import datetime
from botocore.exceptions import ClientError
from typing import Optional, Dict, List, Tuple
from model import get_model, LLM
import re

def signal_handler(signum, frame):
    """Handle termination signals by raising KeyboardInterrupt"""
    raise KeyboardInterrupt()

dotenv.load_dotenv('.env')
print_progress = None

def download_from_s3(s3_client, bucket: str, prefix: str, local_dir: str):
    """
    Download all files from S3 maintaining directory structure
    """
    try:
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # List objects in the S3 prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                # Skip if the object is a directory (ends with '/')
                if obj['Key'].endswith('/'):
                    continue
                    
                # Get relative path by removing prefix
                relative_path = obj['Key'][len(prefix):].lstrip('/')
                if not relative_path:  # Skip if it's the prefix itself
                    continue
                    
                # Create local directory structure
                local_path = os.path.join(local_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                try:
                    # Download file
                    s3_client.download_file(bucket, obj['Key'], local_path)
                except IsADirectoryError:
                    # Skip if we encounter a directory
                    continue
                except Exception as download_error:
                    print(f"Error downloading {obj['Key']}: {str(download_error)}")
                    continue
                    
    except Exception as e:
        raise Exception(f"Error downloading from S3: {str(e)}")

def clear_s3_prefix(s3_client, bucket: str, prefix: str):
    """
    Delete all objects under the specified S3 prefix while keeping the folder structure
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            objects_to_delete = []
            for obj in page.get('Contents', []):
                # Skip the prefix/folder itself
                if obj['Key'] == prefix or obj['Key'].endswith('/'):
                    continue
                objects_to_delete.append({'Key': obj['Key']})
                
            if objects_to_delete:
                s3_client.delete_objects(
                    Bucket=bucket,
                    Delete={'Objects': objects_to_delete}
                )
    except Exception as e:
        raise Exception(f"Error clearing S3 prefix: {str(e)}")

def upload_to_s3(s3_client, local_dir: str, bucket: str, prefix: str):
    """
    Upload local directory to S3 maintaining directory structure
    """
    try:
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                # Calculate relative path
                relative_path = os.path.relpath(local_path, local_dir)
                # Create S3 key with prefix
                s3_key = f"{prefix.rstrip('/')}/{relative_path}"
                s3_client.upload_file(local_path, bucket, s3_key)
    except Exception as e:
        raise Exception(f"Error uploading to S3: {str(e)}")

def get_system_details() -> str:
    """
    Retrieve basic details about the host environment.
    This information will help the model to decide which dependencies need to be installed.
    """
    try:
        uname = subprocess.run("uname -a", shell=True, capture_output=True, text=True).stdout.strip()
    except Exception as e:
        uname = f"Unknown OS: {e}"

    details = f"System Information:\n{uname}"
    return details

def clean_status(table, project_id:str):
    #Remove workflow and file_descriptions
    table.update_item(
        Key={
            'project_id': project_id
        },
        UpdateExpression='REMOVE #wf, #fd',
        ExpressionAttributeNames={
            '#wf': 'workflow',
            '#fd': 'file_descriptions'
        }
    )

def update_message(table, project_id:str, message:str, progress:str=None, verbose:bool=True):
    # Add the key of message status to the project_id key table
    update_expression = 'SET #msg = :message, #lu = :lastupdate'
    expression_names = {
        '#msg': 'message',
        '#lu': 'lastupdate'
    }
    expression_values = {
        ':message': message,
        ':lastupdate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if progress is not None:
        update_expression += ', #progress = :progress'
        expression_names['#progress'] = 'progress'
        expression_values[':progress'] = progress

    table.update_item(
        Key={
            'project_id': project_id
        },
        UpdateExpression=update_expression,
        ExpressionAttributeNames=expression_names,
        ExpressionAttributeValues=expression_values
    )

    if verbose:
        print(message)
def update_workflow_status(table, project_id:str, stage:str, current:str):
    # Add the key of message status to the project_id key table
    update_expression = 'SET #wf.#stage = :stage, #wf.#curr = :current, #lu = :lastupdate'
    expression_names = {
        '#wf': 'workflow',
        '#stage': 'stage',
        '#curr': 'curr',  # Changed from 'current' to avoid reserved keyword
        '#lu': 'lastupdate'
    }
    expression_values = {
        ':stage': stage,
        ':current': current,
        ':lastupdate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # First ensure the workflow map exists
    try:
        table.update_item(
            Key={
                'project_id': project_id
            },
            UpdateExpression='SET #wf = if_not_exists(#wf, :empty_map)',
            ExpressionAttributeNames={'#wf': 'workflow'},
            ExpressionAttributeValues={':empty_map': {}}
        )
    except ClientError:
        pass  # Ignore if workflow already exists

    # Then update the workflow status
    table.update_item(
        Key={
            'project_id': project_id
        },
        UpdateExpression=update_expression,
        ExpressionAttributeNames=expression_names,
        ExpressionAttributeValues=expression_values
    )

def update_workflow_detail(table, project_id:str, workflow_detail:dict):
    # First ensure the workflow map exists
    try:
        table.update_item(
            Key={
                'project_id': project_id
            },
            UpdateExpression='SET #wf = if_not_exists(#wf, :empty_map)',
            ExpressionAttributeNames={'#wf': 'workflow'},
            ExpressionAttributeValues={':empty_map': {}}
        )
    except ClientError:
        pass  # Ignore if workflow already exists

    # Then update the workflow detail
    update_expression = 'SET #wf.#det = :detail'
    expression_names = {
        '#wf': 'workflow',
        '#det': 'detail'
    }
    expression_values = {
        ':detail': workflow_detail
    }

    table.update_item(
        Key={
            'project_id': project_id
        },
        UpdateExpression=update_expression,
        ExpressionAttributeNames=expression_names,
        ExpressionAttributeValues=expression_values
    )

def update_file_descriptions(table, project_id:str, file_descriptions:dict):
    # Add the key of message status to the project_id key table
    # file_descriptions key
    update_expression = 'SET #fd = :file_descriptions'
    expression_names = {
        '#fd': 'file_descriptions'
    }
    expression_values = {
        ':file_descriptions': file_descriptions
    }

    table.update_item(
        Key={
            'project_id': project_id
        },
        UpdateExpression=update_expression,
        ExpressionAttributeNames=expression_names,
        ExpressionAttributeValues=expression_values
    )

def strip_code_markdown(text: str) -> str:
    """
    Extracts and returns only the code content inside a markdown code fence.
    If markdown code fences are present (e.g., 
    ```
    xxx.language
    <content>
    ```
    ),
    only the content between the first opening and closing code fence is returned. 
    Otherwise, returns the original text with leading and trailing whitespace removed.
    """
    pattern = r"```(?:\S+)?\s*\n([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return text

    
def confirm(prompt: str, interactive: bool = False) -> bool:
    """Ask the user to confirm before running a command if interactive mode is enabled."""
    if not interactive:
        return True
    response = input(f"{prompt} (y/n): ")
    return response.strip().lower().startswith("y")

def run_shell_command(cmd: str, skip_confirmation: bool = False, interactive: bool = False) -> Tuple[int, str, str]:
    """Run a shell command after asking for confirmation if in interactive mode."""
    print("Command to run: {cmd}")
    if not skip_confirmation and interactive:
        if not confirm("Do you want to run this command?", interactive=interactive):
            print("Command canceled by user.")
            return 0, "", ""
    
    stripped_cmd = cmd.strip()

    # Handle source/. commands
    if stripped_cmd.startswith(("source ", ". ")):
        try:
            # Extract the file path
            _, script_path = stripped_cmd.split(None, 1)
            script_path = script_path.strip()
            
            if not os.path.exists(script_path):
                err_msg = f"Source file not found: {script_path}"
                print(err_msg)
                return 1, "", err_msg
            
            # Use a login shell to source the file and export the environment
            shell_cmd = ["/bin/bash", "-l", "-c", f"source {script_path} && env"]
            proc = subprocess.run(shell_cmd, capture_output=True, text=True)
            
            if proc.returncode == 0:
                # Update environment variables from the sourced file
                for line in proc.stdout.splitlines():
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
                return 0, proc.stdout, ""
            else:
                return proc.returncode, proc.stdout, proc.stderr
                
        except Exception as e:
            err_msg = f"Failed to source file: {e}"
            print(err_msg)
            return 1, "", err_msg
    
    # Handle export commands properly
    if stripped_cmd.startswith("export "):
        try:
            # Remove 'export' and split on '='
            _, var_assignment = stripped_cmd.split(" ", 1)
            var_name, var_value = var_assignment.split("=", 1)
            var_name = var_name.strip()
            var_value = var_value.strip().strip('"').strip("'")
            
            # Handle PATH specially to append/prepend
            if var_name == "PATH":
                current_path = os.environ.get("PATH", "")
                if ":" in var_value:
                    # Handle PATH="$PATH:/new/path" format
                    var_value = var_value.replace("$PATH", current_path)
                os.environ["PATH"] = var_value
            else:
                os.environ[var_name] = var_value
                
            print(f"Exported environment variable {var_name}={var_value}")
            return 0, var_value, ""
        except Exception as e:
            err_msg = f"Failed to export variable: {e}"
            print(err_msg)
            return 1, "", err_msg
        
    def limit_resources():
        try:
            # Get system memory info
            total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            
            # Allow up to 75% of total system memory
            memory_limit = int(total_memory * 0.8)
            
            # Set memory limit (RLIMIT_AS: address space limit)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Set nice value to slightly reduce priority but not too much
            # Higher numbers = lower priority (range: -20 to 19)
            os.nice(5)
        except Exception as e:
            print(f"Warning: Failed to set resource limits: {e}")


    # For command chains, execute in a single shell to maintain context
    if "&&" in stripped_cmd:
        # Use a login shell to ensure proper environment
        shell_cmd = ["/bin/bash", "-l", "-c", stripped_cmd]
        proc = subprocess.run(shell_cmd, capture_output=True, text=True, preexec_fn=limit_resources)
        return proc.returncode, proc.stdout, proc.stderr

    # Handle cd commands
    if stripped_cmd.startswith("cd "):
        try:
            target_dir = stripped_cmd.split(" ", 1)[1].strip()
            os.chdir(target_dir)
            print(f"Changed working directory to: {os.getcwd()}")
            return 0, "", ""
        except Exception as e:
            err_msg = f"Failed to change directory: {e}"
            print(err_msg)
            return 1, "", err_msg
            
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, preexec_fn=limit_resources)
    return proc.returncode, proc.stdout, proc.stderr

def run_command_with_resolution(model:LLM, cmd: str, skip_confirmation: bool = False) -> Tuple[int, str, str]:
    """Run a shell command and handle errors with AI assistance."""
    resolution_history = []
    attempted_suggestions = set()  # Track attempted suggestions to avoid loops
    original_dir = os.getcwd()  # Store the original directory
    
    while True:
        code, out, err = run_shell_command(cmd, skip_confirmation)
        if code != 0:
            model.start_session()
            print_progress(f"Error executing command: {err}")

            while True:
                suggestion = resolve_error(model, cmd, resolution_history, original_dir)
                
                #clean /tmp/aicoder folder contents but keep tmp folder
                if os.path.exists("/tmp/aicoder"):
                    shutil.rmtree("/tmp/aicoder")
                    os.makedirs("/tmp/aicoder")

                if not suggestion or suggestion in attempted_suggestions:
                    model.clean_session()
                    raise RuntimeError("No new suggestions available. Giving up.\nHistory: \n"+"\n".join([
                        f"\tResolution Attempt: {entry['attempted_resolution']}\n"
                        f"\tFailed Command: {entry['failed_command']}\n"
                        f"\tCommand Output: {entry['command_output']}\n"
                        f"\tError Message: {entry['error_message']}\n"
                        for entry in resolution_history
                    ]))
                
                attempted_suggestions.add(suggestion)
                cmd_list = suggestion.split(" && ")
                success = True
                
                
                for new_cmd in cmd_list:
                    formatted_list = [f"**'{cmd}'**" if cmd == new_cmd else f"'{cmd}'" for cmd in cmd_list]
                    print_progress(f"Running suggestion: [{', '.join(formatted_list)}]")

                    new_code, new_out, new_err = run_shell_command(new_cmd, skip_confirmation)
                    
                    if new_code != 0:
                        if os.getcwd() != original_dir:
                            os.chdir(original_dir)
                            print(f"Restored working directory to: {original_dir}")
                        resolution_history.append({
                            "attempted_resolution": suggestion,  # Full resolution suggestion
                            "failed_command": new_cmd,          # Specific command that failed
                            "command_output": new_out,         # Output from the command
                            "error_message": new_err           # Error message from the command
                        })
                        success = False
                        break
                
                if success:
                    model.clean_session()
                    return 0, new_out, ""  # Return success if all commands succeeded
        else:
            return code, out, err
            
def resolve_error(model:LLM, command: str, resolution_history: List, original_directory: str = None) -> str:
    """
    Analyze command history and suggest alternative commands to achieve the original goal.
    Returns a string of commands separated by ' && ' or None if no solution found.
    
    Examples:
    - If 'flutter' command not found -> Install Flutter SDK
    - If 'npm' not found -> Install Node.js
    - If Python package missing -> Install via pip
    - If directory doesn't exist -> Create directory
    """
    # Format command history for the model
    history_text = "\n".join([
        f"\tResolution Attempt: {entry['attempted_resolution']}\n"
        f"\tFailed Command: {entry['failed_command']}\n"
        f"\tCommand Output: {entry['command_output']}\n"
        f"\tError Message: {entry['error_message']}\n"
        for entry in resolution_history
    ])

    print(f"Resolution History:\n{history_text}")
    
    system_info = get_system_details()
    attempted_commands = set()
    for entry in resolution_history:
        # Add both full resolutions and individual commands to the set
        attempted_commands.add(entry['attempted_resolution'])
        if ' && ' in entry['attempted_resolution']:
            attempted_commands.update(entry['attempted_resolution'].split(' && '))

    
    prompt = (
        "You are a software developer. Analyze the command history and suggest commands to fix the error.\n\n"
        "CONTEXT:\n"
        f"Original Command: {command}\n"
        f"Resolution History:\n{history_text}\n"
        f"Previously Attempted Commands: {', '.join(attempted_commands)}\n"
        f"System Info:\n{system_info}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze error and determine required setup/dependencies\n"
        "2. Suggest commands to:\n"
        "   - Install latest stable versions of tools/SDKs/runtimes\n"
        "   - Set up environment variables and configurations\n"
        "   - Create required directories/files\n"
        "   - Fix permissions or configuration\n"
        "3. Command inclusion rules:\n"
        "   - ALWAYS separate environment setup from original work using && \n"
        f"   - For installations/setup, format as: <setup> && cd {original_directory} && {command}\n"
        "   - For alternative approaches, do not include original command\n"
        "4. Directory handling rules:\n"
        "   - Use standard installation paths when available (e.g., /usr/local for binaries)\n"
        "   - Use /tmp/aicoder ONLY for temporary downloads and builds\n"
        "   - Never use current directory for downloads or installations\n"
        f"   - ALWAYS return to {original_directory} before running original command\n"
        "5. Return commands in markdown format:\n"
        "```bash\n"
        "<setup> && cd \"original_dir\" && <original_command>\n"
        "```\n"
        "6. If no solution possible, return empty code block\n\n"
        "RULES:\n"
        "1. Always follow official installation methods:\n"
        "   - Use official package repositories when possible\n"
        "   - Curl from official sources (https only)\n"
        "   - Verify downloads with checksums/gpg when available\n"
        "   - Add necessary signing keys for repositories\n"
        "2. Handle common scenarios:\n"
        "   - Command not found -> Install tool && cd back && run original command\n"
        "   - Missing dependencies -> Install deps && cd back && run original command\n"
        "   - Permission errors -> Fix permissions && cd back && run original command\n"
        "   - Missing paths -> Create paths && cd back && run original command\n"
        "   - Invalid command -> Suggest correct alternative (no original command)\n"
        "3. Use best practices for package managers:\n"
        "   - apt: apt-get update before install\n"
        "   - brew: brew update before install\n"
        "   - npm: use -g for global tools, --save-dev for dev dependencies\n"
        "   - pip: use virtual environments when appropriate\n"
        "4. Version management:\n"
        "   - Use version managers when available (nvm, pyenv, etc.)\n"
        "   - Install LTS/stable versions by default\n"
        "   - Add required repositories for latest versions\n"
        "5. Directory and installation operations:\n"
        "   - Use standard paths for installations:\n"
        "     * /usr/local/bin for individual executables\n"
        "     * /usr/local/lib for individual libraries\n"
        "     * /opt for complete packages/SDKs (Flutter, Java, etc)\n"
        "     * /usr/share for architecture-independent files\n"
        "   - When building/extracting in /tmp/aicoder:\n"
        "     * NEVER export or use binaries directly from /tmp/aicoder\n"
        "     * For complete packages (Flutter, Java, etc):\n"
        "       - Extract directly to final location when possible: tar -xf file.tar.gz -C /opt/\n"
        "       - If direct extraction not possible, move immediately after extraction\n"
        "       - Configure Git trust if needed: git config --global --add safe.directory /opt/<package_dir>\n"
        "       - Then export PATH or other environment variables\n"
        "     * For individual tools:\n"
        "       - Extract/build only needed files\n"
        "       - Move directly to final location\n"
        "     * Clean up IMMEDIATELY after each operation:\n"
        "       - Remove archives after extraction: rm file.tar.gz\n"
        "       - Remove build files after compilation\n"
        "       - Clean /tmp/aicoder between operations\n"
        "   - Use make install or equivalent when available\n"
        f"   - Format for complete packages: cd /tmp/aicoder && curl -L <url> | tar xz -C /opt/ && rm -rf /tmp/aicoder/* && git config --global --add safe.directory /opt/<package_dir> && export PATH=/opt/<package_dir>/bin:$PATH && cd {original_directory} && <work>\n"
        f"   - Format for individual tools: cd /tmp/aicoder && curl -L <url> | tar xz && cd <build_dir> && make && mv bin/* /usr/local/bin/ && mv lib/* /usr/local/lib/ && cd .. && rm -rf * && cd {original_directory} && <work>\n"
        "6. Avoid commands that were already attempted\n"
        "7. NO explanation text - only code block\n"
    )
    
    response = model.get_model_response(prompt)
    
    suggestion = strip_code_markdown(response).strip()
    
    # If suggestion is empty or just whitespace, return None
    if not suggestion or suggestion.isspace():
        return None
        
    return suggestion
    

def load_specification(input_dir:str, file_name: str) -> str:
    """Load technical specification from the given path."""

    file_path = os.path.join(input_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_name} not found at {input_dir}")
    
    with open(file_path, "r") as f:
        return f.read()

def determine_workflow(model:LLM, spec: str) -> dict:
    """Determine the project workflow and store it in self.workflow."""
    workflow = determine_work_flow(model, spec)
    workflow["generate"] = ["code generation"]
    return workflow
    
def print_workflow_summary(workflow) -> None:
    setup_workflow = workflow.get("setup", [])
    compile_workflow = workflow.get("compile", [])
    run_workflow = workflow.get("run", [])

    # Confirm with the workflow
    print("\nSetup Workflow:")
    for i, cmd in enumerate(setup_workflow):
        print(f"{i + 1}. {cmd}")
    print("\nCode generation here.")
    print("\nCompile Workflow:")
    for i, cmd in enumerate(compile_workflow):
        print(f"{i + 1}. {cmd}")
    print("\nRun Workflow:")
    for i, cmd in enumerate(run_workflow):
        print(f"{i + 1}. {cmd}")

def determine_work_flow(model:LLM, spec: str) -> dict:
    """
    Determine the project workflow based on the provided technical specification and host environment details.

    This function builds a prompt for an expert DevOps engineer to generate a sequential list of shell commands
    required to fully initialize the project environment starting from a blank slate. The generated commands
    must cover all necessary steps, organized into three phases:
    
    - "setup"   : Commands to initialize the project environment and install dependencies.
    - "compile" : Command to check the generated code for syntax errors or other issues.
    - "run"     : Command to execute the code after the verification phase.
    
    The expected output is a strict JSON object with the following structure:
    
        {
        "setup": ["command1", "command2", "command3", ...],
        "compile": ["command_for_compilation_or_error_checking"],
        "run": ["command_to_execute_the_code"]
        }
        
    Args:
        model: An instance providing the 'get_model_response' method for querying the language model.
        spec (str): The technical specification for the project.

    Returns:
        dict: JSON object with keys "setup", "compile", and "run".
    """
    # Retrieve host environment details.
    env_details = get_system_details()

    # Build the prompt with clear instructions and the updated requirements.
    prompt = (
        "You are an expert DevOps engineer. Your task is to generate shell commands for project initialization.\n\n"
        "RESPONSE FORMAT:\n"
        '{\n'
        '    "setup": ["command1", "command2"],\n'
        '    "compile": ["command1"],\n'
        '    "run": ["command1"]\n'
        '}\n\n'
        "REQUIREMENTS:\n"
        "1. Return ONLY a valid JSON object with exactly these three keys: 'setup', 'compile', and 'run'\n"
        "2. Each key must map to an array of shell command strings\n"
        "3. Include commands that:\n"
        "   - setup: MUST include project initialization steps:\n"
        "     * Project creation commands (tools already installed):\n"
        "       - Flutter: 'flutter create <name>', 'flutter pub get'\n"
        "       - Go: 'go mod init <name>', 'go mod tidy'\n"
        "       - Angular: 'ng new <name> --routing --style=scss --skip-install', 'npm install'\n"
        "       - React: 'npx create-react-app <name>', 'npm install'\n"
        "       - Vue: 'vue create <name> --no-git --default', 'npm install'\n"
        "       - Python: 'pip install <required-packages>'\n"
        "       - Node.js: 'npm init -y', 'npm install <dependencies>'\n"
        "     * Configuration files:\n"
        "       - .gitignore setup\n"
        "       - Linter configurations\n"
        "       - Environment files\n"
        "     * Directory structure creation\n"
        "     * Asset/resource initialization\n"
        "     * Database setup if required\n"
        "   - compile: ONLY include static analysis and syntax checking\n"
        "     * Example: 'flutter analyze' for Flutter projects\n"
        "     * Example: 'pylint' or 'mypy' for Python\n"
        "     * Example: 'eslint' for JavaScript\n"
        "     * NO test commands (e.g., NO 'flutter test', 'pytest', 'jest')\n"
        "   - run: Execute the verified code\n"
        "4. DO NOT include any explanation or introduction text\n\n"
        "COMMAND GUIDELINES:\n"
        "1. Chain related commands with '&&'\n"
        "2. Use exact version numbers for stability\n"
        "3. Include error checking where possible\n"
        "4. Add necessary environment variables\n"
        "5. Handle dependencies in correct order\n"
        "6. Use appropriate flags for non-interactive execution\n"
        "7. Include cleanup commands if needed\n\n"
        f"Technical Specification:\n```{spec}```\n\n"
        f"Host Environment Details:\n```{env_details}```"
    )

    print_progress("Querying model for project initialization, compilation checking, and execution commands...")
    response = model.get_model_response(prompt)

    # Attempt to parse the JSON response.
    try:
        workflow = json.loads(response)
    except Exception as e:
        print("Failed to parse JSON response:", e)
        # Try to extract a JSON object from the raw response using regex.
        import re
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                workflow = json.loads(json_match.group(0))
            except Exception as ee:
                raise ValueError("Extracted JSON is invalid.") from ee
        else:
            raise ValueError("No valid JSON found in model response.")
    
    # Validate that the required keys are present.
    required_keys = {"setup", "compile", "run"}
    if not required_keys.issubset(set(workflow.keys())):
        raise ValueError(f"JSON response missing required keys. Found keys: {list(workflow.keys())}")
    
    return workflow

def get_full_file_content(model:LLM, spec, description, file_path, file_structure) -> str:
    """
    Generate (and update) the content for a file as plain text (not JSON) based on a brief file description.
    The AI model is instructed to return only the code.
    If the generated code is complete, it should NOT append the special identifier '[[CONTINUE]]'.
    If the generated code is incomplete it should append the special identifier '[[CONTINUE]]'
    on a new line at the very end.
    When this marker is detected, this function will repeatedly prompt the model
    for continuation until the marker is no longer present.
    
    In this updated version, the follow-up prompt instructs the model to return modifications
    as a unified diff (like GitHub changes). The diff is then applied to the current content.
    """
    marker = "[[CONTINUE]]"
    full_content = ""
    model.start_session()
    
    # Initial prompt.
    initial_prompt = (
        f"You are a professional software developer. Your task is to generating code for {file_path}.\n\n"
        "CONTEXT:\n"
        f"1. File Purpose: {description}\n"
        f"2. Project Structure:\n{file_structure}\n"
        f"3. Technical Requirements:\n{spec}\n\n"
        "INSTRUCTIONS:\n"
        "1. Generate complete, production-ready code\n"
        "2. Include ALL necessary code - no skipping\n"
        "3. No ellipsis or placeholders\n"
        "4. Only use specified libraries\n"
        "5. Return code in markdown format:\n"
        f"   ```<language>\n"
        "   <your code here>\n"
        "   ```\n"
        f"6. If incomplete, append '{marker}' inside the code block\n\n"
        "IMPORTANT:\n"
        "- Generate ALL implementation details\n"
        "- Include ALL imports and helper functions\n"
        "- Maintain consistent style\n"
        "- Code must be fully functional\n"
        "- NO explanatory text outside code block\n"
    )

    response = model.get_model_response(initial_prompt)
    cleaned_response = strip_code_markdown(response).rstrip()
    need_continue = False
    if (cleaned_response.endswith(marker)):
        need_continue = True
        cleaned_response = cleaned_response[:-len(marker)].strip()

    full_content += cleaned_response
    
    # Write the initial generated content into the file.
    with open(file_path, "w") as fw:
        fw.write(full_content)
    print_progress(f"Initial content written to {file_path}")

    # Continue prompting if the current content ends with the special marker.
    while need_continue:

        code_lines = full_content.splitlines()
        context_snippet = "\n".join(code_lines[-10:]) if len(code_lines) >= 10 else full_content
        
        # Build follow-up prompt using unified diff instructions.
        followup_prompt = (
            f"Continue generating code for {file_path}.\n\n"
            "CONTEXT:\n"
            f"1. Last Generated Section:\n{context_snippet}\n"
            f"2. File Purpose: {description}\n\n"
            "INSTRUCTIONS:\n"
            "1. Continue EXACTLY where the code left off\n"
            "2. Generate ALL remaining code\n"
            "3. Return code in markdown format:\n"
            f"   ```<language>\n"
            "   <your code here>\n"
            "   ```\n"
            "4. Return empty code block if complete\n"
            f"5. If incomplete, append '{marker}' inside code block\n\n"
            "IMPORTANT:\n"
            "- Maintain consistent style\n"
            "- Include ALL implementation details\n"
            "- No placeholders or TODOs\n"
            "- NO explanatory text outside code block\n"
        )

        print_progress(f"Continuing content generation for file: {file_path} with current content. Waiting for follow-up generation.")
        response = model.get_model_response(followup_prompt)
        cleaned_response = strip_code_markdown(response).rstrip()
        
        need_continue = False
        if (cleaned_response.rstrip().endswith(marker)):
            need_continue = True
            cleaned_response = cleaned_response[:-len(marker)]
        
        # If no modifications are provided, exit the loop.
        if cleaned_response == "":
            break
        
        full_content += cleaned_response
        
        # Write the updated content to the file.
        with open(file_path, "w") as fw:
            fw.write(full_content)
        print_progress(f"File {file_path} updated with modifications from follow-up prompt.")
    
    model.clean_session()

    return full_content

def generate_code_files(model:LLM, spec, update_filedescription=None) -> dict:
    """
    Uses the provided technical specification and AI model to generate code files, now including a brief description
    for each file to guide content generation. It works in two phases:
    
    1. Query the model for a list of files that should be generated along with a brief description of what each file should contain.
        The model returns a JSON object in the following format:
        {
            "files": [
                {"path": "file1.py", "description": "Description of file1 content"},
                {"path": "dir/file2.py", "description": "Description of file2 content"},
                ...
            ]
        }
    2. For each file:
        - Ensure its directory and file exist (all paths are relative to the output folder).
        - Use the provided description as context to prompt the AI to generate the file content in parts if necessary.
        - Accumulate the file content and ask for confirmation before writing it to disk.
    """
    def build_directory_tree(start_dir="."):
        tree_lines = []
        for root, dirs, files in os.walk(start_dir):
            level = root.replace(start_dir, "").count(os.sep)
            indent = " " * 4 * level
            tree_lines.append(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for file in files:
                tree_lines.append(f"{subindent}{file}")
        return "\n".join(tree_lines)


    project_structure = build_directory_tree(".")

    # Query the model for the list of files to generate along with brief descriptions.
    list_prompt = (
        "You are an expert software developer tasked with analyzing a technical specification and existing project structure.\n\n"
        "CONTEXT:\n"
        f"Current Project Structure:\n{project_structure}\n\n"
        "INSTRUCTIONS:\n"
        "1. Return ONLY a valid JSON object with exactly this structure:\n"
        "{\n"
        '    "files": [\n'
        '        {\n'
        '            "path": "file1.py",\n'
        '            "description": "Detailed description including:\n'
        '                - Main purpose and responsibilities\n'
        '                - Key classes/functions to implement\n'
        '                - Important algorithms or business logic\n'
        '                - External dependencies and imports\n'
        '                - Data structures and models\n'
        '                - Error handling requirements\n'
        '                - Integration points with other files",\n'
        '            "status": "new|exists|modify"\n'
        '        }\n'
        "    ]\n"
        "}\n\n"
        "REQUIREMENTS:\n"
        "1. Project Structure Analysis:\n"
        "   - Examine current structure from setup phase\n"
        "   - For Flutter: check lib/, test/, etc.\n"
        "   - For Angular: check src/app/, components/\n"
        "   - For React: check src/, public/\n"
        "   - For Python: check package structure\n"
        "2. File Status Determination:\n"
        "   - 'new': File doesn't exist and needs creation\n"
        "   - 'modify': File exists but needs modifications\n"
        "3. Path Requirements:\n"
        "   - Use relative paths from project root\n"
        "   - Follow framework conventions:\n"
        "     * Flutter: lib/screens/, lib/widgets/\n"
        "     * Angular: src/app/components/, src/app/services/\n"
        "     * React: src/components/, src/hooks/\n"
        "     * Python: src/, tests/\n"
        "4. Description Requirements:\n"
        "   - Core functionality and purpose\n"
        "   - Required classes, methods, functions\n"
        "   - Data structures and types\n"
        "   - Dependencies and imports:\n"
        "     * ONLY use dependencies from specification\n"
        "     * NO additional third-party packages\n"
        "     * Standard library imports must be justified\n"
        "   - Error handling requirements\n"
        "   - Integration points with other files\n"
        "   - Important algorithms or business logic\n"
        "   - Configuration or environment requirements\n"
        "5. Framework-Specific Handling:\n"
        "   - Flutter:\n"
        "     * Respect existing lib/main.dart\n"
        "     * Add screens to lib/screens/\n"
        "     * Add widgets to lib/widgets/\n"
        "   - Angular:\n"
        "     * Check existing components\n"
        "     * Follow module structure\n"
        "     * Respect routing setup\n"
        "   - React:\n"
        "     * Maintain component hierarchy\n"
        "     * Check for existing hooks\n"
        "     * Respect routing structure\n"
        "   - Python:\n"
        "     * Follow package structure\n"
        "     * Maintain module organization\n"
        "6. NO explanation text outside JSON\n"
        "7. Dependencies MUST be from spec\n\n"
        f"Technical Specification:\n{spec}\n"
    )
    print_progress("Querying model for list of files to generate with descriptions...")
    response = model.get_model_response(list_prompt)
    try:
        result = json.loads(response, strict=False)
        files_list = result.get("files", [])
        if not files_list:
            print("No files listed in the model response.")
            return
    except Exception as e:
        print("Error parsing file list from model response:", e)
        print("Model response was:")
        print(response)
        return

    # Build a directory tree from the file paths.
    def build_tree(paths):
        tree = {}
        for path in paths:
            parts = path.split(os.sep)
            current = tree
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
        return tree

    def tree_to_string(tree, prefix=""):
        output = ""
        keys = sorted(tree.keys())
        for i, key in enumerate(keys):
            is_last = i == len(keys) - 1
            connector = "└── " if is_last else "├── "
            output += prefix + connector + key + "\n"
            if tree[key]:
                new_prefix = prefix + ("    " if is_last else "│   ")
                output += tree_to_string(tree[key], new_prefix)
        return output

    all_paths = [file_item["path"] if isinstance(file_item, dict) else file_item for file_item in files_list]
    project_structure = tree_to_string(build_tree(all_paths))
    print("Plan Project Directory Structure:")
    print(project_structure)

    file_descriptions = {
        item.get("path") if isinstance(item, dict) else item: 
        item.get("description", "No description provided.") if isinstance(item, dict) else "No description provided."
        for item in files_list 
        if isinstance(item, dict) and item.get("path") or not isinstance(item, dict)
    }

    if (update_filedescription is not None):
        update_filedescription(file_descriptions)


    # Process each file individually.
    for i, file_item in enumerate(files_list):
        if isinstance(file_item, dict):
            file_path = file_item.get("path")
            description = file_item.get("description", "No description provided.")
        else:
            file_path = file_item
            description = "No description provided."
        
        if not file_path:
            print("Skipping file with missing path.")
            continue

        print_progress(f"Processing file: {file_path}")

        # Ensure the directory exists (paths are relative to the output folder)
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            mkdir_cmd = f"mkdir -p {directory}"
            run_command_with_resolution(model, mkdir_cmd, skip_confirmation=True)

        # Create the file if it does not exist already.
        if not os.path.exists(file_path):
            touch_cmd = f"touch {file_path}"
            run_command_with_resolution(model, touch_cmd, skip_confirmation=True)

        print_progress(f"{i + 1}/{len(files_list)} - Generating content for file: {file_path}.\nDescription: {description}")
        # Generate file content using repeated prompts if necessary, based on the augmented description.
        full_content = get_full_file_content(model, spec, description, file_path, project_structure)


        if full_content is None:
            print_progress(f"Failed to generate content for {file_path}. Skipping.")
            continue
    
    return file_descriptions


def compile_handle(model:LLM, spec, commands, file_descriptions) -> None:
    """
    Handles compilation and resolution of errors.

    For each compile command in the provided list, the command is run. If the command fails,
    a prompt is sent to the model to extract the error details into a JSON array.
    For each error (including file name, line, column, and message), the function:
    - Reads the current content of the file (if found).
    - Creates a directory tree string based on the current project structure.
    - Prompts the model (with the file content, directory structure, and error details) to return a correction suggestion.
    - Applies the suggestion, either as a unified diff patch or executes suggested shell commands.
    The function then re-runs the compile command and repeats error handling until compilation succeeds.
    """
    import os
    import json


    # Helper: Build a string representation of the directory tree.
    def build_directory_tree(start_dir="."):
        tree_lines = []
        for root, dirs, files in os.walk(start_dir):
            level = root.replace(start_dir, "").count(os.sep)
            indent = " " * 4 * level
            tree_lines.append(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for file in files:
                tree_lines.append(f"{subindent}{file}")
        return "\n".join(tree_lines)


    project_structure = build_directory_tree(".")

    num_run = 0
    total_errors = 0
    resolved_errors = 0

    # Process each compilation command.
    for cmd in commands:
        isFirst = False
        while True:
            print_progress(f"Running compilation command: {cmd}")
            code, out, err = run_shell_command(cmd, skip_confirmation=isFirst)
            isFirst = True
            if code == 0:
                print_progress(f"Compilation command '{cmd}' executed successfully.")
                break  # Move on to the next compile command.
            else:
                print_progress(f"Error executing compilation command '{cmd}':\n{err}")
                num_run += 1
                # Build a prompt to ask the model to extract error details.
                prompt_errors = (
                    "You are analyzing compilation errors to generate a dependency-ordered fix plan.\n\n"
                    "CONTEXT:\n"
                    f"1. Command: {cmd}\n"
                    f"2. Output: {out}\n"
                    f"3. Error: {err}\n"
                    f"4. Project Structure:\n{project_structure}\n\n"
                    "RESPONSE FORMAT:\n"
                    "```json\n"
                    "[\n"
                    "  {\n"
                    '    "file": "filename.ext",\n'
                    '    "message": "Concise error description with cross-references"\n'
                    "  }\n"
                    "]\n"
                    "```\n\n"
                    "RULES:\n"
                    "1. Group all errors by file\n"
                    "2. Order by dependency (dependencies first)\n"
                    "3. Include missing implementations\n"
                    "4. Combine related errors\n"
                    "5. Maximum 50 critical errors\n"
                    "6. Skip redundant errors\n\n"
                    "MESSAGE GUIDELINES:\n"
                    "1. Include cross-file references\n"
                    "2. Specify line numbers\n"
                    "3. Group similar errors\n"
                    "4. Be specific\n\n"
                    "EXAMPLE ERROR INPUT:\n"
                    "main.py:10: SyntaxError: missing semicolon\n"
                    "main.py:11: SyntaxError: missing semicolon\n"
                    "utils.py:15: NameError: name 'process_data' not defined\n"
                    "utils.py:20: ImportError: cannot import 'validate'\n\n"
                    "EXAMPLE JSON OUTPUT:\n"
                    "[\n"
                    "  {\n"
                    '    "file": "helpers.py",\n'
                    '    "message": "Implement validate function (needed by utils.py)"\n'
                    "  },\n"
                    "  {\n"
                    '    "file": "utils.py",\n'
                    '    "message": "ImportError: cannot import \'validate\'; Implement process_data (needed by main.py)"\n'
                    "  },\n"
                    "  {\n"
                    '    "file": "main.py",\n'
                    '    "message": "SyntaxError: missing semicolon on lines 10-11"\n'
                    "  }\n"
                    "]\n\n"
                    "IMPORTANT:\n"
                    "- Return ONLY valid JSON array\n"
                    "- Include ALL dependency files\n"
                    "- NO explanatory text\n"
                    "- NO markdown formatting\n"
                )
                print_progress("Querying model for error categorization...")
                error_response = model.get_model_response(prompt_errors)
                try:
                    error_list = json.loads(error_response)
                except Exception as e:
                    print_progress(f"Failed to parse error categorization from model response: {e}")
                    error_list = []
                
                if not error_list:
                    print_progress("No errors were extracted by the model. Cannot proceed with error handling.")
                    continue

                total_errors += len(error_list)

                files_need_to_fix = set(error_item.get("file") for error_item in error_list)
                model.start_session()
                # Process each reported error.
                for i, error_item in enumerate(error_list):
                    meta_info = f"{num_run} run, {i + 1} / {len(error_list)} error, {total_errors} total errors, {resolved_errors} resolved errors"

                    description = file_descriptions.get(error_item.get("file"))
                    if not description:
                        description = "There is no description for this file."
                    error_item["description"] = description
                    syntax_error_handling(model, error_item, spec, project_structure, files_need_to_fix, meta_info)

                    resolved_errors += 1

                model.clean_session()
                print_progress("Re-running the compilation command after applying corrections...")


def syntax_error_handling(model:LLM, error_item, spec, project_structure, files_need_to_fix, meta_info:str="") -> None:
    import re
    import os

    file_path = error_item.get("file")
    if not file_path:
        return
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            current_content = f.read()
    else:
        print(f"File {file_path} not found. Skipping error handling for this error.")
        return

    marker = "[[CONTINUE]]"

    print_progress(f"{meta_info}\nFixing compilation error in {file_path}...\nDescription: {error_item.get('description')}\nError: {error_item.get('message')}")
    # model.start_session()
    # Build the initial prompt.
    # The model is instructed to inspect the error message, file content, and project structure.
    # It should decide whether extra file context is needed by returning a directive in the form:
    #   # REQUIRED_FILES: file1.py, file2.py
    # If no extra context is needed, it should return only the corrected code.
    prompt = (
        f"You are a professional software engineer, fixing a compilation error in {file_path}.\n\n"
        "CONTEXT:\n"
        f"1. Error Message: {error_item.get('message')}\n"
        f"2. File Purpose: {error_item.get('description')}\n"
        f"3. Current Code:\n{current_content}\n"
        f"4. Project Structure:\n{project_structure}\n"
        f"5. Technical Requirements:\n{spec}\n\n"
        "RESPONSE FORMAT:\n"
        "Option 1 - Request Additional Files:\n"
        "```text\n"
        "# REQUIRED_FILES: file1.py, file2.py\n"
        "```\n\n"
        "Option 2 - Provide Fixed Code:\n"
        f"```<language>\n"
        "<complete fixed code>\n"
        "```\n\n"
        "Option 3 - Package/SDK Version Update:\n"
        "```shell\n"
        "# PACKAGE_UPDATE\n"
        "command1 && command2 && command3\n"
        "```\n\n"
        "INSTRUCTIONS:\n"
        "1. If additional context needed, use Option 1\n"
        "2. If code needs fixing, use Option 2\n"
        "3. If package/SDK version needs updating, use Option 3\n"
        f"4. For long responses, append '{marker}' inside code block\n\n"
        "IMPORTANT:\n"
        "- For Option 3:\n"
        "  * Chain multiple commands with &&\n"
        "  * Provide exact version constraints\n"
        "  * Include all dependent updates\n"
        "  * Order updates correctly (dependencies first)\n"
        "  * Use package manager commands (npm, pip, pub)\n"
        "  * Example: update SDK && update package && fix dependencies\n"
        "- Maintain all functionality\n"
        "- Generate ALL code - no skipping\n"
        "- No explanations outside code blocks\n"
        "- Follow exact response format\n"
    )

    code_so_far = ""
    # This flag ensures that the REQUIRED_FILES directive is only acted upon once.
    initial_step = True

    while True:
        response = model.get_model_response(prompt)

        if initial_step:
            output = strip_code_markdown(response).rstrip()
            
            if (response == output):
                print("Model return with invalid format. Skipping...")
                return

            if output.startswith("# PACKAGE_UPDATE"):
                update_commands = output.replace("# PACKAGE_UPDATE", "").strip().split(" && ")
                for cmd in update_commands:
                    cmd = cmd.strip()
                    if not cmd:
                        continue
                    print_progress(f"Executing command for compilation error: {cmd}")
                    code, out, err = run_command_with_resolution(model, cmd)
                    if code != 0:
                        print_progress(f"Package update failed: {err}")
                        return
                print_progress("Finsished command updates.")
                return

            # Look for a REQUIRED_FILES directive only in the first response.
            directive_match = re.match(r"#\s*REQUIRED_FILES:\s*(.*)", output)
            if directive_match:
                print_progress(f"Addtional files requested for context. {directive_match.group(1)}")
                extra_files_list = [fname.strip() for fname in directive_match.group(1).split(",") if fname.strip()]

                #Check if any file requested is in the files_need_to_fix set
                # for fname in extra_files_list:
                #     if fname not in files_need_to_fix:
                #         print(f"AdditionalFile {fname} is in the files_need_to_fix. Will not fix {file_path} for now...")
                #         return

                extra_context = ""
                for fname in extra_files_list:
                    if os.path.exists(fname):
                        with open(fname, "r") as ef:
                            extra_context += f"Content of {fname}:\n{ef.read()}\n\n"
                    else:
                        extra_context += f"Content of {fname}: File not found.\n\n"
                prompt = (
                    f"You are a professional software engineer, fixing a compilation error in {file_path}.\n\n"
                    "CONTEXT:\n"
                    f"1. Error Message: {error_item.get('message')}\n"
                    f"2. File Purpose: {error_item.get('description')}\n"
                    f"3. Additional Files:\n{extra_context}\n"
                    f"4. Current File:\n{current_content}\n"
                    f"5. Project Structure:\n{project_structure}\n"
                    f"6. Technical Requirements:\n{spec}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Provide complete fixed code in markdown format:\n"
                    f"   ```<language>\n"
                    "   <your code here>\n"
                    "   ```\n"
                    f"2. If response is incomplete, append '{marker}' inside code block\n\n"
                    "IMPORTANT:\n"
                    "- Maintain all functionality\n"
                    "- Generate ALL code - no skipping\n"
                    "- No explanations outside code block\n"
                    "- Follow exact response format\n"
                )
                # Continue to get the proper code with the extra context.
                continue
            else:
                # No directive means the response is already code.
                output = strip_code_markdown(response).rstrip()
                code_so_far += output
                # Do not check for extra files from subsequent responses.
                initial_step = False
        else:
            # For follow-up generations, ignore any accidental REQUIRED_FILES directive.
            # Remove such a directive if present.
            response = re.sub(r"^#\s*REQUIRED_FILES:\s*.*\n?", "", response, count=1)
            output = strip_code_markdown(response).rstrip()
            code_so_far += output

        # Check if the accumulated code ends with the continue marker.
        if code_so_far.endswith(marker):
            # Remove the marker.
            code_so_far = code_so_far[:-len(marker)].rstrip()
            # Use the last 10 lines of the current code as context.
            code_lines = code_so_far.splitlines()
            context_snippet = "\n".join(code_lines[-10:]) if len(code_lines) >= 10 else code_so_far
            prompt = (
                f"Continue fixing compilation error in {file_path}.\n\n"
                "CONTEXT:\n"
                f"1. Error Message: {error_item.get('message')}\n"
                f"2. Last Generated Section:\n{context_snippet}\n\n"
                "INSTRUCTIONS:\n"
                "1. Continue EXACTLY where you left off\n"
                "2. Provide code in markdown format:\n"
                f"   ```<language>\n"
                "   <your code here>\n"
                "   ```\n"
                "3. Return empty code block if complete\n"
                f"4. If incomplete, append '{marker}' inside block\n\n"
                "IMPORTANT:\n"
                "- Maintain consistent style\n"
                "- Generate ALL code - no skipping\n"
                "- No explanations outside code block\n"
            )
            continue
        else:
            break

    # Write the final corrected content to the file.
    with open(file_path, "w") as f:
        f.write(code_so_far)
    print_progress(f"Regeneration complete. Updated content written to {file_path}.")
    # model.clean_session()
    return code_so_far


def main(project_id: str, file_name: str = None):
    signal.signal(signal.SIGTERM, signal_handler)  # Handle Docker container termination
    signal.signal(signal.SIGINT, signal_handler)   # Handle keyboard interrupt

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'us-east-2')
    )
    dynamodb = boto3.resource(
        'dynamodb',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'us-east-2')
    )
    bucket = os.environ.get('S3_BUCKET_NAME')
    table = dynamodb.Table('neon-ai-v1_process')

    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{project_id}/",
            MaxKeys=1
        )
        if 'Contents' not in response:
            raise Exception(f"Project folder {project_id} does not exist in S3")
    except ClientError as e:
        raise Exception(f"Error accessing S3: {str(e)}")
    
    try:
        response = table.get_item(
            Key={
                'project_id': project_id
            }
        )
        if 'Item' not in response:
            raise Exception(f"Project {project_id} does not exist in the database")
    except ClientError as e:
        raise Exception(f"Error accessing DynamoDB: {str(e)}")
    
    s3_input_prefix = f"{project_id}/analyze/output"
    s3_output_prefix = f"{project_id}/coder/output"

    input_directory = os.path.join(os.path.dirname(__file__), "input")
    output_directory = os.path.join(os.path.dirname(__file__), "output")

    def _update_message(message:str, status:str = None):
        update_message(table, project_id, message, status)

    global print_progress
    print_progress = _update_message
    try:

        shutil.rmtree(input_directory, ignore_errors=True)
        shutil.rmtree(output_directory, ignore_errors=True)

        os.makedirs(input_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)

        if not project_id:
            raise Exception("PROJECT_ID environment variable is not set.")
        if not file_name:
            raise Exception("FILE_NAME environment variable is not set.")
        
        clean_status(table, project_id)
        
        _update_message(f"Syncing input file (Directory: {s3_input_prefix})...", "preparing")
        download_from_s3(s3_client, bucket, s3_input_prefix, input_directory)

        model = get_model(api_key=os.getenv("ANTHROPIC_API_KEY"), model_type='claude')

        #Start here
        os.chdir(output_directory)

        spec = load_specification(input_dir=input_directory, file_name=file_name)

        workflow = determine_work_flow(model, spec)
        update_workflow_detail(table, project_id, workflow)

        print_workflow_summary(workflow)

        _update_message("Step 1: Project initialization...", "setup")
        total_setup = len(workflow["setup"])
        for i, cmd in enumerate(workflow["setup"], 1):
            update_workflow_status(table, project_id, "setup", cmd)
            run_command_with_resolution(model, cmd)

        _update_message("Step 2: Code generation...", "generate")
        update_workflow_status(table, project_id, "generate", "code generation")

        def _update_file_descriptions(file_descriptions):
            update_file_descriptions(table, project_id, file_descriptions)

        file_descriptions = generate_code_files(model, spec, _update_file_descriptions)

        _update_message("Step 3: Code compilation...", "compile")
        total_compile = len(workflow["compile"])
        for i, cmd in enumerate(workflow["compile"], 1):
            update_workflow_status(table, project_id, "compile", cmd)
            compile_handle(model, spec, [cmd], file_descriptions)
            
        update_workflow_status(table, project_id, "finish", "finish")
        _update_message(f"Finish", f"finished")
    except KeyboardInterrupt as e:
        update_workflow_status(table, project_id, "terminate", "terminated manually")
        _update_message(f"Finish manually", f"terminated")
    except Exception as e:
        update_workflow_status(table, project_id, "error", f"error: {str(e)}")
        _update_message(f"finish with error: {str(e)}", f"error")
        raise
    finally:
        # Only perform when the output directory is not empty
        if os.path.exists(output_directory) and os.path.isdir(output_directory) and len(os.listdir(output_directory)) > 0:
            _update_message(f"Cleaning up...")
            # Clear S3 output prefix
            clear_s3_prefix(s3_client, bucket, s3_output_prefix)

            _update_message(f"Uploading output (Directory: {s3_output_prefix})...")
            # Always try to upload output if it exists, even if there was an error
            if os.path.exists(output_directory) and os.path.isdir(output_directory):
                try:
                    upload_to_s3(s3_client, output_directory, bucket, s3_output_prefix)
                except Exception as e:
                    print(f"Error uploading output to S3: {str(e)}")
                    raise
            _update_message("Successfully uploaded output to cloud")



if __name__ == "__main__":
    main(project_id=os.getenv("PROJECT_ID"), file_name=os.getenv("FILE_NAME"))