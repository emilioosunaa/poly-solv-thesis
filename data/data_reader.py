import fitz
import pandas as pd
import re

file_path = 'Polymer-Solvent Interaction Parameter.pdf'
pdf_document = fitz.open(file_path)
pages = range(3, 24)

def is_polymer(line):
    return bool(re.search(r'poly|cellulose', line, re.IGNORECASE))

def is_number_or_range(s):
    # Matches numbers and ranges, including negative values for both ends
    s = s.strip()
    return bool(re.match(r'^-?\d+(\.\d+)?(\s*to\s*-?\d+(\.\d+)?)?$', s))

def is_reference(s):
    return '[' in s and ']' in s

def is_empty(s):
    return not s.strip()

def is_table_header(line):
    return any(x in line.lower() for x in ['solvent', 'temperature', 'volume', 'fraction', 'references'])

def normalize_minus(s):
    return (
        s.replace('\x01', '-')     # Control char from PDF extraction
    )

data = []
current_polymer = None
current_solvent = None
buffer = []

for page_num in pages:
    page = pdf_document.load_page(page_num - 1)
    lines = page.get_text("text").split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        line = normalize_minus(line)  # Fix any weird minus signs right away

        if is_empty(line) or is_reference(line) or is_table_header(line):
            i += 1
            continue

        if is_polymer(line):
            current_polymer = line
            current_solvent = None
            buffer = []
            i += 1
            continue

        # Update current_solvent if line is not number/range and not polymer
        if not is_number_or_range(line) and not is_polymer(line):
            current_solvent = line
            buffer = []
            i += 1
            continue

        # Add line to buffer (might be values for temp, vol_frac, chi)
        buffer.append(line)
        if len(buffer) > 3:
            buffer.pop(0)

        # Check if buffer is ready (contains 3 value lines)
        if (
            len(buffer) == 3
            and all(is_number_or_range(x) for x in buffer)
            and current_polymer
            and current_solvent
        ):
            temp, vol_frac, chi = buffer
            # Normalize again just in case (paranoia, for values split off from lines)
            temp = normalize_minus(temp)
            vol_frac = normalize_minus(vol_frac)
            chi = normalize_minus(chi)
            data.append({
                'Polymer': current_polymer,
                'Solvent': current_solvent,
                'Temperature (°C)': temp,
                'Volume fraction, φ2': vol_frac,
                'Interaction Parameter χ': chi,
            })
            buffer = []

        i += 1

df = pd.DataFrame(data)
df.to_csv('polymer_solvent_interaction_parameters.csv', index=False)
print(f"Extracted {len(df)} rows to polymer_solvent_interaction_parameters.csv")
