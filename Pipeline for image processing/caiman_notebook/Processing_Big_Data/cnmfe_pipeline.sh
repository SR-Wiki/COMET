# (文件名应为00.tif ----------66.tif)
#最终结果会保存在save_dir/COMET_patch_merged_all中

python revised_cnmfe.py --src_dir "path/to/your/full/session/data" --split_save_dir "where/to/save/the/split/patches" --patch_size 320 --overlap_pix 20 --save_dir "where/to/save/the/final/merged/results" 

