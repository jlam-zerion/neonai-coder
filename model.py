import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

class LLM(ABC):
    """
    Abstract base class for Language Learning Models
    """
    @abstractmethod
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM with an optional API key
        
        Args:
            api_key (Optional[str]): API key for authentication
        """
        self.api_key = api_key
        self._session = None
        self._conversation_history = []

    def get_conversation_history(self) -> List[str]:
        """
        Get the conversation history
        """
        return self._conversation_history
    
    @abstractmethod
    def start_session(self) -> None:
        """
        Start a new chat session, clearing previous conversation history
        """
        self._session = True
        self._conversation_history = []
    
    @abstractmethod
    def get_model_response(self, 
                            prompt: str, 
                            files: Optional[Union[str, List[str]]] = None,
                            verbose: bool = False,
                            logging: bool = False,
                            gemini_config: Optional[dict] = None) -> str:
        """
        Get a response from the model based on file contents and a prompt
        
        Args:
            prompt (str): Prompt to guide the model's response
            files (Optional[Union[str, List[str]]]): Single file path or list of file paths to analyze
        
        Returns:
            str: Model's generated response
        """
        pass
    
    @abstractmethod
    def clean_session(self) -> None:
        """
        Close the current chat session and clear conversation history
        """
        self._conversation_history = []
        self._session = None

class Gemini(LLM):
    """
    Gemini LLM implementation
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini model
        
        Args:
            api_key (Optional[str]): Google AI API key
        """
        super().__init__(api_key)
        
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai package: pip install google-generativeai")
        
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=self.api_key)
        self._genai = genai
        self._model = genai.GenerativeModel('gemini-1.5-flash')
    
    def start_session(self) -> None:
        """
        Start a new Gemini chat session
        """
        super().start_session()
        self._session = self._model.start_chat(history=[])
    
    def get_model_response(self, 
                       prompt: str, 
                       files: Optional[Union[str, List[str]]] = None, 
                       verbose: bool = False,
                       logging: bool = False,
                       gemini_config: Optional[dict] = None) -> str:
        """
        Get a response from Gemini model.
        
        Args:
            prompt (str): Prompt to guide the model's response.
            files (Optional[Union[str, List[str]]]): Single file path or list of file paths to analyze.
            verbose (bool): If True, outputs the model request and response to stdout.
            logging (bool): If True, logs the model request and response into a file.
            gemini_config (Optional[dict]): Configuration options for Gemini API.
        
        Returns:
            str: Gemini's generated response.
        """
        def log_message(message: str):
            if verbose:
                print(message)
            if logging:
                full_path = os.path.normpath( os.path.join(__file__, "..", "gemini_response.log"))
                with open(full_path, "a") as log_file:
                    log_file.write(message + "\n")
                    
        if verbose or logging:
            log_message("=====")
            log_message("Given prompt:")
            log_message(prompt)
            log_message("Getting Gemini response...")
        
        if gemini_config is None:
            gemini_config = {}
        
        # If files are provided, read their contents and prepend to the prompt
        if files:          
            file_contents = []
            for file_path in files:
                with open(file_path, 'r') as f:
                    file_contents.append(f.read())
                # Prepend file contents to the prompt
            full_prompt = "File contents:\n" + "\n---\n".join(file_contents) + f"\n\nPrompt: {prompt}"
        else:
            full_prompt = prompt
        
        full_response = ""
        # Use session if started; otherwise, use default model.
        if self._session:
            response = self._session.send_message(full_prompt, generation_config=gemini_config, stream=True)
        else:
            response = self._model.generate_content(full_prompt, generation_config=gemini_config, stream=True)

        if verbose or logging:
            log_message("----")
            log_message("Response: ")
            
        for chunk in response:
            full_response += chunk.text
            if verbose or logging:
                log_message(chunk.text)
        
        # Store conversation history.
        self._conversation_history.append({
            'prompt': full_prompt,
            'response': full_response
        })
        
        if verbose or logging:
            log_message("=====")
        
        return full_response
    
    def clean_session(self) -> None:
        """
        Close the Gemini chat session
        """
        super().clean_session()
        # Additional cleanup specific to Gemini if needed

class Claude(LLM):
    """
    Claude LLM implementation without conversation history
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude model
        
        Args:
            api_key (Optional[str]): Anthropic API key
        """
        super().__init__(api_key)
        
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")

        try:
            import chardet
        except ImportError:
            raise ImportError("Please install chardet package: pip install chardet")

        if not self.api_key:
            raise ValueError("Claude API key is required")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        # Dictionary to cache file processing results.
        # Each key will be a file path and its value will be a dict containing:
        #   mtime      : last modified time when processed.
        #   processed  : content processed from the file.
        #   summary    : (optional) summarized content if the original is too long.
        self._file_cache = {}

    def start_session(self) -> None:
        """
        No-op method for session start
        """
        super().start_session()
    
    def get_model_response(self, 
                        prompt: str, 
                        files: Optional[Union[str, List[str]]] = None, 
                        verbose: bool = False,
                        logging: bool = False,
                        claude_config: Optional[dict] = None) -> str:
        """
        Get a response from Claude model without maintaining conversation history.
        If the combined prompt (including file contents) is too long,
        each file's content will be summarized to reduce token count while preserving context.
        
        Args:
            prompt (str): Prompt to guide the model's response.
            files (Optional[Union[str, List[str]]]): Single file path or list of file paths to analyze.
            verbose (bool): If True, outputs the model request and response to stdout.
            logging (bool): If True, logs the model request and response into a file.
            claude_config (Optional[dict]): Configuration options for Claude API.
        
        Returns:
            str: Claude's generated response.
        """
        import mimetypes
        import chardet
        from tqdm import tqdm  # Using tqdm for progress bars

        def log_message(message: str):
            if verbose:
                print(message)
            if logging:
                full_path = os.path.normpath(os.path.join(__file__, "..", "claude_response.log"))
                with open(full_path, "a") as log_file:
                    log_file.write(message + "\n")
                    
        if verbose or logging:
            log_message("=====")
            log_message("Given prompt:")
            log_message(prompt)
            log_message("Getting Claude response...")

        if claude_config is None:
            claude_config = {}
                
        # Set default configuration if not provided.
        config = {
            'model': claude_config.get('model', 'claude-3-opus-20240229'),
            'max_tokens': claude_config.get('max_tokens', 4096),
            'temperature': claude_config.get('temperature', 0.5)
        }
            
        # Helper to estimate tokens via simple word count.
        def _estimate_tokens(text: str) -> int:
            return len(text.split())
            
        # Process file contents if files are provided.
        if files:
            # Ensure files is a list.
            if isinstance(files, str):
                files = [files]
                
            # Store tuples of (file_path, processed_content).
            file_entries = []
            for file_path in tqdm(files, desc="Processing file contents", unit="file", disable=not verbose):
                try:
                    current_mtime = os.path.getmtime(file_path)
                    cache_entry = self._file_cache.get(file_path)
                    if cache_entry is not None and cache_entry.get('mtime') == current_mtime:
                        processed_content = cache_entry.get('processed')
                        file_entries.append((file_path, processed_content))
                        continue

                    file_size = os.path.getsize(file_path)
                    mime_type, _ = mimetypes.guess_type(file_path)
                        
                    # Skip files that are too large (e.g., over 1MB).
                    if file_size > 1_000_000:
                        processed_content = f"[File {file_path} skipped: Size {file_size} bytes exceeds 1MB limit]"
                        self._file_cache[file_path] = {'mtime': current_mtime, 'processed': processed_content}
                        file_entries.append((file_path, processed_content))
                        continue
                        
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                        result = chardet.detect(raw_data)
                        encoding = result['encoding'] or 'utf-8'
                        
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                            
                        # Truncate very long file contents.
                        if len(content) > 10_000:
                            content = content[:10_000] + "\n\n[... content truncated]"
                            
                        processed_content = (f"File: {file_path}\n"
                                            f"Size: {file_size} bytes\n"
                                            f"MIME Type: {mime_type or 'unknown'}\n"
                                            f"Encoding: {encoding}\n---\n" + content)
                    except UnicodeDecodeError:
                        processed_content = f"[File {file_path} skipped: Unable to decode with {encoding} encoding]"
                        
                    self._file_cache[file_path] = {'mtime': current_mtime, 'processed': processed_content}
                    file_entries.append((file_path, processed_content))
                
                except (PermissionError, FileNotFoundError) as e:
                    processed_content = f"[Error reading {file_path}: {str(e)}]"
                    self._file_cache[file_path] = {'mtime': None, 'processed': processed_content}
                    file_entries.append((file_path, processed_content))
                
            file_contents = [entry[1] for entry in file_entries]
            file_contents_text = "File contents:\n" + "\n---\n".join(file_contents)
            MAX_FILE_CONTENT_TOKENS = 5000
                
            # Summarize file contents if the token count exceeds the limit.
            if _estimate_tokens(file_contents_text) > MAX_FILE_CONTENT_TOKENS:
                log_message("File contents exceed token limit; summarizing each file's content to reduce length.")
                summarized_contents = []
                for file_path, content in tqdm(file_entries, desc="Summarizing file contents", unit="file", disable=not verbose):
                    cache_entry = self._file_cache.get(file_path)
                    if cache_entry is not None and 'summary' in cache_entry:
                        summary_text = cache_entry['summary']
                    else:
                        try:
                            summary_prompt = (
                                "Summarize the following content in a detailed manner, capturing all technical aspects "
                                "and preserving crucial context:\n\n" + content
                            )
                            summary_response = self.client.messages.create(
                                model=config['model'],
                                max_tokens=config['max_tokens'],
                                temperature=config['temperature'],
                                messages=[{"role": "user", "content": summary_prompt}]
                            )
                            summary_text = summary_response.content[0].text.strip()
                        except Exception as summary_error:
                            log_message(f"Error summarizing content from a file: {summary_error}")
                            summary_text = content
                        if cache_entry is not None:
                            self._file_cache[file_path]['summary'] = summary_text
                    summarized_contents.append(summary_text)
                file_contents_text = "Summaries of file contents:\n" + "\n---\n".join(summarized_contents)
                
            full_prompt = file_contents_text + f"\n\nPrompt: {prompt}"
        else:
            full_prompt = prompt
            
        if self._session is None:
            messages = [{"role": "user", "content": full_prompt}]
        else:
            messages = self._conversation_history + [{"role": "user", "content": full_prompt}]
            
        try:
            response = self.client.messages.create(
                model=config['model'],
                max_tokens=config['max_tokens'],
                temperature=config['temperature'],
                messages=messages
            )
            full_response = response.content[0].text
        except Exception as e:
            log_message(f"Error in Claude API call: {e}")
            return ""
            
        if verbose or logging:
            log_message("----")
            log_message("Response: ")
            log_message(full_response)
            log_message("=====\n")
            
        if self._session is not None:
            self._conversation_history.append({"role": "user", "content": full_prompt})
            self._conversation_history.append({"role": "assistant", "content": full_response})
            
        return full_response
    
    def clean_session(self) -> None:
        """
        No-op method for session cleanup
        """
        super().clean_session()


def get_model(api_key: Optional[str] = None, model_type: str = "gemini") -> LLM:
    """
    Get a model instance based on the provided API key
    
    Args:
        api_key (Optional[str]): API key for authentication
        model_type (str): Type of model to use
    
    Returns:
        LLM: An instance of the LLM class
    """
    if not api_key:
        # Try to get API key from environment variables
        if model_type == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
        elif model_type == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(f"API key for {model_type} is required")
    
    if model_type == "gemini":
        return Gemini(api_key)
    elif model_type == "claude":
        return Claude(api_key)
    else:
        raise ValueError(f"Unknown model type: {model_type}")