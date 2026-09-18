import pandas as pd
import os

aptos_path = r"C:\Users\iPC\Downloads\raw_aptos"

def analyze_split(csv_name, img_dir_name):
    csv_path = os.path.join(aptos_path, csv_name)
    img_dir = os.path.join(aptos_path, img_dir_name, img_dir_name)
    
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # APTOS usually has 'id_code' and 'diagnosis'
    id_col = 'id_code' if 'id_code' in df.columns else df.columns[0]
    label_col = 'diagnosis' if 'diagnosis' in df.columns else df.columns[1]
    
    print(f"--- Analysis for {csv_name} ---")
    print(f"Total rows in CSV: {len(df)}")
    print("Class distribution:")
    print(df[label_col].value_counts().sort_index())
    
    # Check images
    if os.path.exists(img_dir):
        img_files = os.listdir(img_dir)
        print(f"Total files in {img_dir_name}: {len(img_files)}")
        
        # Check correspondence
        csv_ids = set([str(x) + '.png' for x in df[id_col]] + [str(x) + '.jpg' for x in df[id_col]] + [str(x) for x in df[id_col]])
        dir_ids = set(img_files)
        
        # Exact match logic might vary based on extensions
        # Let's just check how many stems match
        csv_stems = set([str(x).replace('.png','').replace('.jpg','') for x in df[id_col]])
        dir_stems = set([x.rsplit('.', 1)[0] for x in img_files])
        
        missing_in_dir = csv_stems - dir_stems
        missing_in_csv = dir_stems - csv_stems
        
        print(f"Images in CSV but missing in Dir: {len(missing_in_dir)}")
        print(f"Images in Dir but missing in CSV: {len(missing_in_csv)}")
    else:
        print(f"Missing image directory {img_dir}")
    print("\n")

analyze_split('train_1.csv', 'train_images')
analyze_split('valid.csv', 'val_images')
analyze_split('test.csv', 'test_images')
