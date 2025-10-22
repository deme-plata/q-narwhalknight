import re

with open('main.rs', 'r') as f:
    content = f.read()

# Replace all append_log calls with direct property updates
# Pattern: self.ui.append_log(...) -> 
# New: let current_log = self.ui.get_log(); self.ui.set_log(format!("{}{}", current_log, ...));

# Strategy: Add a helper method instead
helper_method = '''
    /// Helper to append text to log
    fn append_to_log(&self, text: &str) {
        let current = self.ui.get_log();
        self.ui.set_log(std::format!("{}{}", current, text).into());
    }
'''

# Find the position after the impl AppState block starts
impl_start = content.find('impl AppState {')
if impl_start != -1:
    # Find the first method after impl
    first_method = content.find('fn new()', impl_start)
    if first_method != -1:
        # Insert helper before first method
        content = content[:first_method] + helper_method + '\n\n    ' + content[first_method:]

# Now replace all append_log calls
content = re.sub(
    r'(\s+)self\.ui\.append_log\((.*?)\);',
    r'\1self.append_to_log(&\2.to_string());',
    content
)

with open('main.rs', 'w') as f:
    f.write(content)

print("✅ Fixed append_log calls")
