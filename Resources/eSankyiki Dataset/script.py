import csv
from bs4 import BeautifulSoup
import os

def clean_text(text):
    """Clean and normalize text"""
    return text.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

def extract_dataset_info(div):
    """Extract dataset information from a catalog-grid div"""
    data = {
        'Title': '',
        'Product': '',
        'Category': '', 
        'Geography': '',
        'Frequency': '',
        'Reference Period': '',
        'Release Date': '',
        'Table No': '',
        'Download URL': '',
        'Data Source': '',
        'Description': ''
    }
    
    # Extract title
    title = div.find('h5', class_='title')
    if title:
        data['Title'] = clean_text(title.get_text())
    
    # Extract metadata from list items
    for li in div.find_all('li'):
        text = clean_text(li.get_text())
        if not text:
            continue
            
        if text.startswith('Product:'):
            data['Product'] = clean_text(text.split(':', 1)[1].split('Category')[0].strip())
        elif text.startswith('Category:'):
            data['Category'] = clean_text(text.split(':', 1)[1].split('Geography')[0].strip())
        elif text.startswith('Geography:'):
            data['Geography'] = clean_text(text.split(':', 1)[1].split('Frequency')[0].strip())
        elif text.startswith('Frequency:'):
            data['Frequency'] = clean_text(text.split(':', 1)[1].split('Reference Period')[0].strip())
        elif text.startswith('Reference Period:'):
            data['Reference Period'] = clean_text(text.split(':', 1)[1].split('Data Source')[0].strip())
        elif text.startswith('Data Source:'):
            data['Data Source'] = clean_text(text.split(':', 1)[1].split('Description')[0].strip())
        elif text.startswith('Description:'):
            data['Description'] = clean_text(text.split(':', 1)[1].split('Release Date')[0].strip())
        elif text.startswith('Release Date:'):
            data['Release Date'] = clean_text(text.split(':', 1)[1].split('Table No.')[0].strip())
        elif text.startswith('Table No.'):
            data['Table No'] = clean_text(text.split('Table No.')[1])
    
    # Extract download URL
    download_link = div.find('a', href=True)
    if download_link and 'download' in download_link.attrs:
        data['Download URL'] = download_link['href']
    
    return data

def extract_all_datasets(html_content):
    """Extract all datasets from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    dataset_divs = soup.find_all('div', class_='catalog-grid')
    return [extract_dataset_info(div) for div in dataset_divs]

def save_to_csv(datasets, filename='datasets.csv'):
    """Save extracted datasets to CSV file"""
    if not datasets:
        print("No datasets found to save.")
        return
    
    fieldnames = datasets[0].keys()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(datasets)
    
    print("Data saved to", filename)

# Example usage:
if __name__ == "__main__":
    for i in range(1,3):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        html_filename = 'WMI'+str(i)+'.html'  # Change this to the actual HTML file name
        html_path = os.path.join(script_dir, html_filename)

        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {html_path}")
            print("Please ensure the HTML file exists in the same directory as this script.")
            exit(1)

        # Extract the base name without extension
        base_name = os.path.splitext(html_filename)[0]
        csv_filename = f"{base_name}_details.csv"
        output_path = os.path.join(script_dir, csv_filename)

        datasets = extract_all_datasets(html_content)

        if datasets:
            save_to_csv(datasets, output_path)
            print(f"Successfully extracted {len(datasets)} clean datasets and saved to {csv_filename}")
        else:
            print("No datasets found in the HTML content")