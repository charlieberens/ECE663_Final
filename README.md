## Geting Started
* Video Link to Slurm guide: https://compsci.capture.duke.edu/Panopto/Pages/Viewer.aspx?id=a803886c-2117-4806-ae99-aaea00d749a7
* Git Link to Slurm guide: https://gitlab.cs.duke.edu/wjs/cs-cluster-talk

* SSH into <net_id>@login.cs.duke.edu
	- Copy over everything you need. I usually just use scp.
	- Do not store large files on the login server! You have a small-ish limit, and if you have VS Code connected remotely it will make your life a living hell. (It has to index every file and will make everything really laggy).
		+ Instead store everything somewhere in the "/usr/project/xtmp/<net_id>" folder. I usually make a simlink to make things easier, ex:  ` sl datasets /usr/project/xtmp/<NET_ID>/datasets`
		+ The project is in my xtmp folder so if you want to manage our current directory make a simlink to my directory using `sl 663Proj ../../../usr/project/xtmp/rnb23/663Proj`
* Using Slurm
	- There are two ways to do it
		+ Interactively: Use this if you want to test the code (don't run code directly on the login server, run this command first)
			* I will generally run "srun --nodes=1 --ntasks-per-node=1  --partition=compsci-gpu --gres=gpu:1 --time=01:00:00 --pty zsh -i"
				- You should switch "zsh" to "bash" if you're using bash/don't have zsh installed
				- You should definitely change these settings to get the correct resources (see below for more info)
		+ Batch jobs: Use this if you want to run one or multiple jobs in the background
			* Run "sbatch <training_script>.sh" (I pasted an example training script below)
			* Here is my script I used to train cat2kat
			
			```bash
            #!/usr/bin/env bash

            #SBATCH --job-name=RunCatsTraining
            #SBATCH --mem=20gb
            #SBATCH --time=48:00:00
            #SBATCH --partition=compsci-gpu
            #SBATCH --gres=gpu:a6000:1
            #SBATCH --output=logs/RunCatsTraining_%j.out   # Where to save the log

            # The following will actually be run.
            ls -a
            eval "$(conda shell.bash hook)" 
            conda init
            sleep 10
            conda activate img2img-turbo
            cd img2img-turbo
            export NCCL_P2P_DISABLE=1
            accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
                --pretrained_model_name_or_path="stabilityai/sd-turbo" \
                --output_dir="output/cyclegan_turbo/cat2kat" \
                --dataset_folder "data/cat2kat" \
                --train_img_prep "no_resize" --val_img_prep "no_resize" \
                --learning_rate="1e-5" --max_train_steps=25000 \
                --train_batch_size=2 --gradient_accumulation_steps=2 \
                --report_to "wandb" --tracker_project_name "cat2kat_1" \
                --enable_xformers_memory_efficient_attention --validation_steps 1000 \
                --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1
			```
## Making a dataset

* **What is ArtiFact:**  
So far we have used [ArtiFact dataset](https://www.kaggle.com/datasets/awsaf49/artifact-dataset) for training. This is very large and full of tons of different real and fake images all labeled and categorized. The downside is they are all 256x256 images and there is a significant lack of quality in the real images. Often time the subject is wrong or nonexistent and many of the labels are nonsense. If we want to use this for stuff past cats and people we will want to spend some time relabeling the dataset. Before trying to work with the dataset I recommend looking through it on kaggle and looking at the metadata to understand how it works and what the labels are, as they are often inconsistent.

* **How to use ArtiFact:**  
I have written a script to pull whatever images you want from the artifact dataset. Its in DataScripts called ArtiFactQuery.py. You should change these parameters in your script based on your needs.

```python
base_path = '../ArtiFact'  # The base directory where the original dataset is located
output_directory = '../cats_fake'  # The directory where the new, filtered dataset should be saved
saved_metadata_path = '../ArtiFact/SavedMetaData.pkl'  # Path to a saved metadata file for faster loading. If it doesn't exist, the script will reindex the metadata, which can be time-consuming.
include_params = {'category': ['cat'], 'target': [1, 2, 3, 4, 5, 6]}  # Specifies which categories, targets, and/or models to include in the new dataset. In this example, only images of 'cat' with target values 1 to 6 (ai generated range) will be included.
exclude_params = {'model': ['pro_gan']}  # Specifies which categories, targets, and/or models to exclude from the new dataset. In this example, images from the 'pro_gan' model will be excluded.
num_images = 10000  # The total number of images to select for the new dataset.
sampling_method = 'distribution'  # Specifies how to sample the images. Options are 'random' or 'distribution'. 'distribution' will aim for an even distribution across the specified parameters in distribution_params.
distribution_params = ['model']  # Defines the parameters to distribute the sampling across. For example, specifying 'model' will ensure an even distribution of images from different models.
```
These parameters are not changed through the command line you need to go into the script and change them yourself.

* **How to make your new dataset work with CycleGAN:**  
  * Once you have your new dataset made you need to split it into training and test sets using the split_val_set.py script.
    * Here are the parameters they should be pretty self explanatory.
      ```python
      parser.add_argument("directory", type=str, help="The path to the directory containing the train set of images.")
      parser.add_argument("val_spec", type=float, help="The proportion or number of images to move to the validation set (0 < val_spec <= 1 for proportion, val_spec >= 1 for specific number).")
      ```
      These parameters are set through the command line.
  * Once you have your data split into a test and training set you need to add fixed prompts for each spaces. These are prompts for the a and b space (real and fake space) of the model to reference and associate with the images. Do this by adding a fixed_prompt_a.txt and a fixed_prompt_b.txt file to the dataset directory. To get a full picture of what the cyclegan dataset should look like read [this](img2img-turbo/docs/training_cyclegan_turbo.md) portion of the img2img-turbo readme.

## How to Train CycleGAN-Turbo

1. **Set Up the Environment**
   - Ensure your dataset is ready for training.
   - Follow the setup instructions in the [CycleGAN-turbo README](img2img-turbo/README.md#getting-started) to create a conda or Python environment.
   - After setting up the environment, manually install the following packages using `pip`:
     - `vision_aided_loss`
     - `wandb`
   - **Note:** These packages might not install correctly if included in the `environment.yaml` file.
   - Configure the environment using Accelerate:
     ```bash
     accelerate config
     ```
     - **Accelerate Configuration Guide**: Below is the exact configuration that I have used for setting up the environment. Please note that it is untested with other configurations, and it’s possible that enabling some advanced features might achieve faster speeds.
        ```
        In which compute environment are you running? 
        - This machine 

        Which type of machine are you using? 
        - No distributed training 

        Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? 
        - NO 

        Do you wish to optimize your script with torch dynamo? 
        - NO 

        Do you want to use DeepSpeed? 
        - NO 

        What GPU(s) (by id) should be used for training on this machine as a comma-separated list? 
        - all 

        Would you like to enable NUMA efficiency? (Currently only supported on NVIDIA hardware). 
        - NO 

        Do you wish to use mixed precision? 
        - fp16
        ```

2. **Modify the Training Script**
   - Use a modified version of the [TrainCat2Kat.sh](img2img-turbo/TrainCat2Kat.sh) script.
   - Refer to the [Training CycleGAN-turbo Documentation](img2img-turbo/docs/training_cyclegan_turbo.md) for detailed instructions on which parameters to modify.
   - Generally, you only need to adjust these CycleGAN-Turbo parameters:
     - `output_dir`
     - `dataset_folder`
     - `tracker_project_name`
   - In the bash script parameters, update:
     - The time allocated to the job
     - The job name
     - The output log file name

3. **Start Training**
   - After making the necessary modifications, save the updated script and run it with SLURM:
     ```bash
     sbatch script.sh
     ```
   - This command will start the training process.

4. **Monitor Training Output**
   - You can track training progress through:
     1. **WandB** (for a more comprehensive view)
     2. **Tail Command** (to display live output in the terminal):
        ```bash
        tail -f logs/output_log_name_jobid#.out
        ```
     - This command displays log output in real-time, as if you were running the script locally.
