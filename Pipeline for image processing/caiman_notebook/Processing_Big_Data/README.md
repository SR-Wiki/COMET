
For large dataset (e.g., 1944x1944x10000+), the original caiman processing pipeline may suffer from memory overflow. Hence, we have made modifications to  fit for this increasing data-size demand.

For motion correction, modify the input_folder and output_folder in motion_correct_batch.py according to your data. After that, choose a large segment_length which fits for your RAM. It is recommended that overlap_size is no less than 20.
```
 # ----------------------------Parameters for batch processing-------------------------------------
    # modify the input_folder and output_folder
    # files in the input_folder should be named as "i.tif", where i is the sequence number
    input_folder = 'E:\COMET-longterm/\\'
    output_folder = 'E:\COMET-longterm/MC\\'
    segment_length = 400     # size of each segment, choose a large one while avoiding memory issues
    overlap_size = 50        # overlap between each segment (>= 20)
```
Then, set the relevant CaImAn parameters in motion_correct_batch.py, and run the code in the terminal or in an IDEA:
```
python motion_correcte_batch.py
```
With motion-corrected files, we can peform CNMF-E algorithm for source extraction:
First, set the CaImAn parameters in source_extraction.py.
Then, run the following code.
```
python revised_cnmfe.py --src_dir "path/to/your/full/session/data" --split_save_dir "where/to/save/the/split/patches" --patch_size 256 --overlap_pix 20 --save_dir "where/to/save/the/final/merged/results"
```
Parameter explanations:

"src_dir" is the output_folder for motion correction,

"split_save_dir" is for patch splitting and can be arbitray

"patch_size" will be set by the same logic as "segment_length", for a 128GB RAM, 256 can be the first attempt

"overlap_size" is for component merging, follows from same parameter of CaImAn CNMF-E.

"save_dir" is where to save the result, and the final merge results will be saved to "save_dir/COMET_patch_merged_all".

As this pipeline will generate large amount of temp files, it is recommended to set "--caiman_data_dir" to direct these files to a folder on a disk with enough storage.

