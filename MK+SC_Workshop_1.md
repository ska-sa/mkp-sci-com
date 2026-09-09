# MK+ Science Commissioning Workshop I


# Workshop Aim

*This workshop aims to provide the necessary information to the MK+ partners’ commissioning team to access the SARAO commissioning machine, execute the relevant software suites, and to organise the structure of the working environment. The idea is to treat this document as a living document throughout the workshop to capture the relevant information, decisions, comments and issues and convert these into a step-by-step handout for the commissioning team and define a base approach to data processing in the [MK+ Github repository](https://github.com/ska-sa/mkp-sci-com) for the entire team.*

## Day 1 

*After that day everybody should be able to work, upload data, and where to find/install/generate software*

### Setup the SARAO account and logging into the SARAO machine

- **SARAO account**  
  - Provide information to **sean@sarao.ac.za** who will send out the agreement  
    - **Title(Ms, Mrs, Mr, Dr, etc)**    
      - **Full Names:**    
      - **Email: (Affiliated Institution email Only)**    
      - **Job Title:**    
      - **Organisation:**    
      - **Name and email address of "Boss"**  “EXTERNAL PARTY PROJECT  AUTHORITY”    
  - **Install VPN and need to change your passwd**   
    - Received e-mail from “SARAO Cybersecurity”  
      - Login username & password  
      - VPN information  
      - (Need to reset your password\!)
     
    - Connect to VPN (e.g. FortiClient)  
  		- FortiClient installation and setup instructions: [FortiClient Installation Instructions - SARAO VPN.pdf](https://drive.google.com/file/d/15I1bP8DOSihx6_PHB_JEzFFYRUz98ZdO/view?usp=drive_link)  
		- If you find FortiClient inconvenient to use.  
  			- Sign in to [https://kat-cpt-vpn.kat.ac.za/vpn-user-portal/](https://kat-cpt-vpn.kat.ac.za/vpn-user-portal/) with your SARAO access.  
  			- Go to Configurations, create and download the VPN profile  
  			- Use the downloaded profile with other VPN clients, such as openVPN [https://openvpn.net/](https://openvpn.net/)
  
  - **Generate a pub key**  
    - ```ssh-keygen -t ed25519 ```  
    - Send Ben (bhugo@sarao.ac.za) the id\_ed25519.pub public key file   
    - (Optional) For security, it is recommended that you use a separate key file for each cluster. In that case, generate the key file with \-f \<outpu\_key\_file\>

		```shell
		ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_bob -C "username@bob.science.kat.ac.za"
		```

    - (Option) Add to your `.ssh/config`

		``` Host bob
  		HostName bob.science.kat.ac.za
  		User <username>  # Change to your username
  		AddKeysToAgent = yes  IgnoreUnknown UseKeychain
  		UseKeychain = yes
  		IdentityFile = ~/.ssh/id_ed25519_bob  # Or point to the private key pair you sent to Ben
		```

  - **Login onto the SARAO MK+ science commissioning machine**  
  	```ssh \-i .ssh/XXXXX YYY@bob.science.kat.ac.za```  
  	(XXXXX being path to your SSH private key, and YYY is your SARAO user name

     
### SARAO machine and the user setup and limitations

- [bob.science.kat.ac.za](http://bob.science.ket.ac.za) (this is the compute machine)  
- work in your user directory  
- 10 TB quota in the home directory  
- another 5 machines for science commissioning will arrive in October 

### Software

- **System**  
  - Singularity  
  - Apptainer [https://apptainer.org/docs/user/latest/](https://apptainer.org/docs/user/latest/)  
  - WSClean  
  - AOFlagger  
  - Carta (organise themself; can be installed in user home)  
  - CASA \- available as system module module load casa/6.6 (need to put the following source in your .bashrc: source /usr/share/modules/init/[profile.sh](http://profile.sh))  
  - Sshfs  
  - `progress`
      
- **Python virtual environment**
  
  We use UV as virtual environment, because its much faster and keeps storage usage minimal with respect to conda or venv or virtualenv.  

  - Set up UV locally – this provides an easy (and fast) way to set up a local Python virtual environment for each user.
    following the instruction in [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).

	```curl -LsSf https://astral.sh/uv/install.sh | sh```
	
	install uv for the different shells:
	      ```source $HOME/.local/bin/env``` (sh, bash, zsh)
	      ```source $HOME/.local/bin/env.fish``` (fish)

	Check your installation 
		```which uv```
		should point to YOUR HOME DIRECTORY/.local/bin/uv
	
	Setup a python version
		```uv venv --python 3.12 py312```

	Activate the python version 	
		```source py312/bin/activate```
	
	Check your python
		```which python```
		should point to your directory where you installed the thing and py312/bin/python
	
  - Install a python package
	assume that the you activate your python virtual environment ```source path/to/env/bin/activate\```  
    - Install packages with ```uv pip install <package1 package2 …>```

  - Packages that you might want to install  
    - Caracal  
    - DDfacet (python, or Cultcargo container)  
    - Radiopadre  
    - Python casacore  
    - Stimela \+ cult-cargo (automatic container support, [Link to the container](https://quay.io/organization/stimela2))  
    - [https://github.com/caracal-pipeline/cult-cargo](https://github.com/caracal-pipeline/cult-cargo)

- **Example of installing and running shadems**   
    
  - In your python environment  
    ```source py312/bin/activate```  
    ```uv pip install shadems```  
    ```which shadems```   
      - points to the py312/bin/shadems  
    ```shadems```   
        
  - Using a container  
    - ```cd ; mkdir CONTAINER\_SHADEMS; cd CONTAINER\_SHADEMS```  
    - ```apptainer pull docker://[quay.io/stimela2/shadems](http://quay.io/stimela2/shadems)```  
    - ```apptainer exec  shadems\_latest.sif shadems```

### Discussion of the organisation (development environment, large workflow execution) and the definition of the work

- **Share of raw datasets**  
  - to prevent everybody to download the same observations, we may want to share the [tar.gz](http://tar.gz) files in /home/MKplus\_SC\_ORG\_DATASETS  
    - Parameter to generate the MS files  

		INSERT PNG
      
    - Currently these files are stored in /home/MKplus\_SC\_ORG\_DATASETS:  
      - 1780929124-sdp-l0\_2026-06-17T15-34-35\_db1.ms.tar.gz
      - 1781430924\_2026-07-03T15-23-56\_4l4.ms.tar.gz  
      - 1783252863\_2026-07-17T12-07-04\_KVG.ms.tar.gz  
          
    - For completeness:  
      - [https://archive.sarao.ac.za/observations/1783252863](https://archive.sarao.ac.za/observations/1783252863)  
      - [https://archive.sarao.ac.za/observations/1781430924](https://archive.sarao.ac.za/observations/1781430924)  
      - [https://archive.sarao.ac.za/observations/1780929124](https://archive.sarao.ac.za/observations/1780929124)

    

- **Data processing and etiquette**

  - For development and testing, reduce the size of the data whenever possible to improve processing speed and efficient use of the SARAO machine.  
  - When evaluating pipelines, investigate and test individual steps of the workflow rather than repeatedly reprocessing the results of previous steps.  
  - Please consider   
  - We may need to consider restriction of the machine if it gets on its knees (Systemd defines all sorts of usage of the hardware CPU & RAM)  
      
      
- **Share of information**  
  - png in the github issue (see an example here: [https://github.com/ska-sa/mkp-heresci-com/issues/9](https://github.com/ska-sa/mkp-sci-com/issues/9)) and that can be tracked link to the file in the Google drive ()  
  - Google drive [Commissioning Data Reduction](https://drive.google.com/drive/folders/1h3TfFNCcSCHMu2PDZAuNcw4B9g5-egx3) 

- Understanding the dataset  (can I reduce the size in a way that the features I want to improve are retained to allow for faster development/processing?)  
- May want to use screen ctrl \+ a and d will detach it for over the weekend processing  
  
## Day 2

*After that day everybody should know how to generate diagnostic plots and to image an observation* 

Presentations on the work done of the first MK+ test observations 178 092 9124,178 143 0924 and 178 325 2863\. 

- [MeerKAT Calibration Documentation](https://skaafrica.atlassian.net/wiki/spaces/ESDKB/pages/1452310549/Calibration) (here you find a good account of the imaging commission needs) and the [MK+ Science commissioning task break down](https://docs.google.com/spreadsheets/d/1fFHCBCgMV0klukg-gqKLh_x1_taXesFLJzURzR8lK84/edit?gid=1133384290#gid=1133384290)

- The output of the the SDP pipeline for each observations:  
    
  - [1780929124](https://archive.sarao.ac.za/s3/1780929124-meerkatreductionproduct/calibration+report-1/calreport1.html?token=eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzg4NDQ0Mjk3LCJwcmVmaXgiOlsiMTc4MDkyOTEyNCIsInMzLzE3ODA5MjkxMjQiXSwiZXhwIjoxNzkzNjI4Mjk3LCJzdWIiOiIwZDYwZDMwNy1jYjI4LTQzNmEtODAyNy0wOTM5OTZhNzZmNmUiLCJzY29wZXMiOlsicmVhZCJdfQ.Oxw8qemLVerXMkVVHa8goTdRz30VxvllHJiQXB3TZRthZTFkySC5DsBrUVy9nd7sY8wTDqZEhDoaqEOzLbpeBg&subtitle=2406.25+Mhz+to+3281.25+Mhz)  
  - [1781430924](https://archive.sarao.ac.za/s3/1781430924-meerkatreductionproduct/calibration+report-1/calreport1.html?token=eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzg4NDQ0MjQ4LCJwcmVmaXgiOlsiMTc4MTQzMDkyNCIsInMzLzE3ODE0MzA5MjQiXSwiZXhwIjoxNzkzNjI4MjQ4LCJzdWIiOiIwZDYwZDMwNy1jYjI4LTQzNmEtODAyNy0wOTM5OTZhNzZmNmUiLCJzY29wZXMiOlsicmVhZCJdfQ.7zD6kDXUXGECcb4VNz-NuesSuOmfwMV_3sHejAVvotYcDFCI0uzB5xws1ENIDGOC_9lKZSijsNt4O4-JfmqJhg&subtitle=856+Mhz+to+1712+Mhz)  
  - [1783252863](https://archive.sarao.ac.za/s3/1783252863-meerkatreductionproduct/calibration+report-1/calreport1.html?token=eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzg4NDQ0MDExLCJwcmVmaXgiOlsiMTc4MzI1Mjg2MyIsInMzLzE3ODMyNTI4NjMiXSwiZXhwIjoxNzkzNjI4MDExLCJzdWIiOiIwZDYwZDMwNy1jYjI4LTQzNmEtODAyNy0wOTM5OTZhNzZmNmUiLCJzY29wZXMiOlsicmVhZCJdfQ.wZ3catYvWjUWx7Bi16fpa71bImsKJbb924mV6ttofTobBk3LH_BELchd7tVmGM2KfWQl0TF04EyvTO1LrzsFXA&subtitle=856+Mhz+to+1712+Mhz)


- Fillipo and Sphe using the workflow in CaraCal:  
    
  - Idea use Caracal as benchmark doing 2 runs: one using only MK telescopes and compare these with a run including the MK+  
  - Software used ([https://github.com/caracal-pipeline/caracal](https://github.com/caracal-pipeline/caracal))  
- Workflow description of the individual steps:  
	- Crosscal workflow description ([https://github.com/ska-sa/mkp-sci-com/issues/13](https://github.com/ska-sa/mkp-sci-com/issues/13))  
	- Self-Cal 2 GC and 3GC and continuum imaging workflow  ([https://github.com/ska-sa/mkp-sci-com/issues/11](https://github.com/ska-sa/mkp-sci-com/issues/11))  
	- Final image a mosaicking ([https://github.com/ska-sa/mkp-sci-com/issues/12](https://github.com/ska-sa/mkp-sci-com/issues/12))  
	- Data quality assessment (MK Only description) [https://github.com/ska-sa/mkp-sci-com/issues/9](https://github.com/ska-sa/mkp-sci-com/issues/9)  
	- Data depository [Continuum images & mosaics](https://drive.google.com/drive/folders/1b6VQ0ZaGy1goLcS3ib7WCLtd2HwjU_C1?usp=share_link)   
	- [Parameter Files](https://drive.google.com/drive/folders/12kXYZ5P4I3XgcCCUKIdWlo9MtjI0J0Jr?usp=share_link)

- Future plans to do that all in Caracal2-Stimela3 (Sphe)  
- Additional Information:  
	- METADATA COLUMN : TELESCOPE\_NAME, needed for the primary beam  
	- [Simms 3.0](https://simms.readthedocs.io/) can:  
	  - Add PB metadata to MS  
	  - Convert apparent sky model (FITS/ASCII) to intrinsic fluxes (with PA rot)  
	  - Simulate intrinsic sky model while applying a primary beam  
	  - MeerKAT/MeerKAT+ are shipped with the package  
	  - See the simms [primary beams and a-terms section of the RTD](https://simms.readthedocs.io/en/latest/concepts/beams.html)  
	  - Explanation of LSM sources  
	- CASA Model Amp Diff is due to Field sources  
	- Less than a \~1 % difference with heterogeneous beams on/off   
  		- This is with 3 M+ antennas, so the effect is expected to be more significant with the full array.  
	- Benchmark tests adapt to all the science commissioning? Benchmark doing always 2 runs: one using only MK telescopes and compare these with a run including the MK+ telescopes.


- Simone calibration and imaging step by step:  
	- Modular workflow (with its scripts and config files) can by found via github ([SJVeronese/mktplus-pipeline](https://github.com/SJVeronese/mktplus-pipeline)) (should work for everybody, for any issue please contact Simone)  
	- Workflow (treat the on-source observation as ‘the calibrator’ and any other field as ‘the target’):  
	  - Extract calibrator visibilities with mstransform  
	  - Standard flagging (autocorr, shadowing, zero-amplitudes, rfi with tfcrop) on the calibrator with flagdata  
	  - KGBKGB crosscal with gaincal and bandpass (using [this](https://skaafrica.atlassian.net/wiki/spaces/ESDKB/pages/1481408634) prescription for the model)  
	  - (optional) inspection of the calibrated visibilities with shadems  
	  - Extract whatever target in the observation and apply on-the-fly calibration tables with applycal  
	  - (optional) inspection of the calibrated target with shadems  
	  - Image the calibrated target with wsclean  
	- Results for observation 1780929124 are [here](https://github.com/ska-sa/mkp-sci-com/issues/10)


- Ben calibration and imaging predicting visibilities:  
	- K,B,G on-axis (source in the phase centre?)  
	- transfer (KGB) to pointings off source  
	- Use DDfacett \- to do directional depended calibration (heterogeneous PB setup, offset of the model and the visibility)  
	- Shift each off-source pointing to source position, check flux density  
	- Predicting visibilities off axis sources,  
	- Configuration presented is for DDfacett   
	- [https://docs.google.com/presentation/d/1HZ50xyrMq5WRhghKp8TwKQsFEbw3OtjjGlO4AesKSPg/edit?usp=sharing](https://docs.google.com/presentation/d/1HZ50xyrMq5WRhghKp8TwKQsFEbw3OtjjGlO4AesKSPg/edit?usp=sharing)   
	- Additional Info   
  		- See comments from Mattieu (some of it need to be addressed with new long observation in polarisation)  
  		- Continuum subtraction is an issue for heterogeneous array.

- Base calibration of the commissioning work (continuum)   
  - Sequence of the basic calibration steps and the cross-calibration  
    - 1GC (basic continuum)  
      - K1, G1, B1, K2, G2, B2 (see Crosscal workflow description; Filippo, Sphe)  
      - KGBKGB (Simone)  
      - KGB only to applied solution (Ben)

- PB questions to Mattieu: 

	1\) At which elevation have you done the PB measurements?   
			It would be roughly 40deg EL.

	1.1) Do you envisage changes at different elevations (20-80 deg)?  
	I do expect differences, but I think the differences shown in Ben's results are likely due to something more significant. This is why I advised ensuring everything is correct first when 	using new software (here DDfacet) by more thorough testing using L-band, with polcal and reference pointing, and wider beam angle coverage than a single sample, but there has been some pushback on this suggestion.  
		

	2\) The PB measurement is based on averaged measurement. Is there also an error beam available?   
	PB is averaged over antennas, and environmental conditions. Although an Errorbeam (relative to array average) can be calculated, it is not yet calculated because the array average is still premature given the limited number of antennas.

	2.1) Would it be possible to provide this in a future release of the PB please? Even if its premature we can start adapting the software etc.  
	Please clarify what you need in terms of errorbeam: E.g. single errorbeam % maximum value as a function of frequency per H and V co-pol beam (or Stokes I beam?), and reference average beam filename used in calculation?

	3\) For dynamic range imaging and accurate modelling of the visibilities individual MK+ Beam might be useful, are those envisaged to be provided in the future ?  
	It is possible to generate individual MK+ beams. When engaging on accurate modeling of visibilities, as a first step one needs to perform a variety of sanity checks to ensure the software reads and implements beams correctly, reference pointing done, polcal done, etc, before jumping at conclusions.

	3.1) Would it be possible to share these also in the next round of your measurements ? We can do the software checks and already start including these kind of tests in parallel of using an averaged PB.  
	It is possible, yes. Possible snags due to limited data include different average elevations per antenna, poorer RFI suppression, and I may be collecting mostly S4 (MKE data) rather than S3 (DVS data), which could cause subband overlap compatibility issues. There may be less S0 data (MKE) available under different environmental conditions, possibly causing overlap issues when trying to extract S3 equivalent beams from S0 and S4. Note I do not collect S3 in MKE, only S0 and S4 to cover the whole S band, but the data is limited especially in S0. In DVS only S3 is collected, also sparsely. My suggestion is to ensure your software can use S4 beams only as a fallback when you observe S3.


- What are the take aways of the first data calibration approaches?  

  - Proof of concept that MK+ datasets can be 1GC calibrated  
  - Generating model visibilities for 2GC and 3GC is possible  
  - Benchmarking in the procedure and that the corrections are applied correctly needs to be done. Need a document reporting on that (input from us all).  

