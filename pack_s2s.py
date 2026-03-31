import os

# CONFIGURATION
output_file = "S2S_flat.txt"
# Folders we don't want to read
ignore_folders = {'.git', '__pycache__', 'node_modules', '.venv', '.DS_Store', 'data', 'logs', 'venv'}
# Files we want to include - ONLY source code files
include_extensions = {'.py', '.yaml', '.yml', '.md', '.toml'}

print(f"🚀 Flattening S2S project into {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    for root, dirs, files in os.walk('.'):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_folders]
        
        for file in files:
            if any(file.endswith(ext) for ext in include_extensions):
                file_path = os.path.join(root, file)
                
                # Write a clear separator so Gemini knows where files start/end
                f.write(f"\n\n{'='*30}\n")
                f.write(f"FILE: {file_path}\n")
                f.write(f"{'='*30}\n\n")
                
                try:
                    with open(file_path, 'rb') as src:
                        raw_content = src.read()
                        # Filter out binary data, keep only printable ASCII and whitespace
                        clean_content = ''.join(char for char in raw_content.decode('utf-8', errors='ignore') 
                                              if ord(char) >= 32 and ord(char) <= 126 or char in '\n\r\t')
                        f.write(clean_content)
                    print(f"✅ Added: {file_path}")
                except Exception as e:
                    f.write(f"[ERROR READING FILE: {e}]")

print(f"\n✨ Success! Your full project is now in {os.path.abspath(output_file)}")
print("You can now upload this file directly to Gemini AI Studio.")
