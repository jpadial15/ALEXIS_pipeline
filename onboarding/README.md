# ALEXIS_pipeline
Automatically Labelled EUV and XRay Incident Solarflare Catalog. Is a pipeline that can re-create the integrated X-Ray flux of the full solar disk into the emission from discrete regions in the Extreme UV. This pipeline is inteaded to identify and catalog solar flares. 

Requirements:
1. Anaconda installed on your system. If you don't have Anaconda installed. Please verify (https://repo.anaconda.com/archive/) and change the wget link to your system. Follow on-screen instructions and install in the default path. 
   ```
   wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

   bash Anaconda3-2024.06-1-Linux-x86_64.sh
   ```
1. Fork ALEXIS_pipeline repository to your own github account. Then,clone your copy of ALEXIS_pipeline to the machine you are working from. A walk through of a similar process can be found here: https://github.com/jpadial15/ai-science-training-series/blob/main/00_introToAlcf/03_githubHomework.md

1. Install conda environment
   ```
   conda env create -f ALEXIS_conda_environment.yml
   
   ```

1. Activate conda environment
   ```
   conda activate aiadl_2
   ```
1. SQLite databases: 
   - Make or download the XRS availability database.
   ```

   ```
  - Make or download the AIA availability database.
  ```

  ```
  - Make or download the HARP DRMS availability database.
  ```
  
  ```
  - Make or download the SXI availability database.
  ```

  ```
  - Make HEK Known flares database.
  ```

  ```

