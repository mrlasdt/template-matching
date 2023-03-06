# python scripts/convert_format_label.py \
#     --in_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_annos/pos04/annotations.xml \
#     --out_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos04.json \
#     --template_dir ./data/templates/POS04


# python scripts/convert_format_label.py \
#     --in_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_annos/pos01/annotations_v0.xml \
#     --out_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos01.json \
#     --template_dir ./data/templates/POS01



# python scripts/convert_format_label.py \
#     --in_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_annos/pos02/annotations.xml \
#     --out_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos02.json \
#     --template_dir ./data/templates/POS02

# python scripts/convert_format_label.py \
#     --in_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_annos/pos03/annotations.xml \
#     --out_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos03.json \
#     --template_dir ./data/templates/POS03

# python scripts/convert_format_label.py \
#     --in_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_annos/pos08/annotations.xml \
#     --out_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos08.json \
#     --template_dir ./data/templates/POS08


# python scripts/convert_format_label.py \
#     --in_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_annos/pos05/annotations.xml \
#     --out_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos05.json \
#     --template_dir ./data/templates/POS05

# python scripts/convert_format_label.py \
#     --in_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_annos/pos06/annotations.xml \
#     --out_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos06.json \
#     --template_dir ./data/templates/POS06

# python scripts/convert_format_label.py \
#     --in_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_annos/pos03/annotations_fields_checkbox.xml \
#     --out_path /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos03_fields_checkbox.json \
#     --template_dir /mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/templates/POS03

for i in 1 2 3 4 5 6 8
do
    python scripts/convert_format_label.py \
        --in_path "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_annos_add_fields/pos0$i/annotations.xml" \
        --out_path "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/add_fields/pos0$i.json" \
        --template_dir "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/templates/POS0$i"
done
