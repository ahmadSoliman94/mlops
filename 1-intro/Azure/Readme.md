[Source video](https://www.youtube.com/watch?v=IXSiYkP23zo&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK)

<br />

# Setup Jupyter Notebook on Azure via SSH using Linux

## Sign in to Azure

Sign in to the [Azure portal](https://portal.azure.com/).

## Create virtual machine
* Enter `virtual machines` in the search or click `Virtual machines` under Azure services. 

<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/azure%20home.PNG?raw=true" alt="enter username" width="700" height="300"/>
</p>

  Under Services, select `Virtual machines`.

<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/azure%20home2.PNG?raw=true" width="700" height="300"/>
</p>

* In the Virtual machines page, select `Create` and then `Azure Virtual machine`. 

<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/az%20vm%20create.PNG?raw=true" width="700" height="300"/>
</p>

  The Create a virtual machine page opens.

* In the Basics tab, under Project details, make sure the correct subscription is 
  selected and then choose to Create new resource group. Enter `<name-of-your-choice>` for the name.
  `mlops` will be used in this tutorial
  
<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/resourcegrp.PNG?raw=true" width="700" height="300"/>
</p>
  
* Under Instance details, enter azurevm for the Virtual machine name, and choose `Ubuntu 20.04 LTS - Gen2` for your Image. 
  Leave the other defaults. The default size and pricing is only shown as an example. 
  Size availability and pricing are dependent on your region and subscription.
  
  The `Standard_E2s_v3 - 2 vcpus, 16 GiB` memory will be used in this tutorial

<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/az%20vm%20create2.PNG?raw=true" alt="enter username" width="700" height="300"/>
</p>  
 
* Set Authentication type to `SSH public key`

  Set Username, Generate new key pair and set key pair name.

<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/pem%20key1.PNG?raw=true" alt="enter username" width="700" height="300"/>
</p> 
 
* Set Inbound port rules
  Allow selected ports and select `'HTTP (80)' and 'SSH (22)'`
  
 <p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/inbound.PNG?raw=true" alt="enter username" width="700" height="300"/>
</p> 
  
  Leave the remaining defaults and then select the `Review + create` button at the bottom of the page.
 
* It will take you to a service pricing webpage. Go through it. 
  When you are done, select Create.

<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/pricing.PNG?raw=true" alt="enter username" width="600" height="600"/>
</p> 
  
* When the Generate new key pair window opens, select Download `private key` and `create resource`. 
  Your key file will be download as `azurevmkey.pem`. Make sure you know where the .pem file was downloaded; 
  you will need the path to it in the next step.

<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/pem_key.PNG?raw=true" width="500" height="300"/>
</p> 

* Afte VM deploys finish, click on `Go to resource`.
 
<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/go%20to%20resource.PNG?raw=true" width="500" height="300"/>
</p> 
  
* Copy Public IP address
 
<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/public-ip.PNG?raw=true" width="500" height="300"/>
</p>   
 
  
## Connect to virtual machine

Create an SSH connection with the VM.

If you are on a Mac or Linux machine, open a Bash prompt. 
If you are on a Windows machine, you can use a PowerShell prompt but in this tutorial, we will use Git bash. 
Click this [link](https://git-scm.com/downloads) 


##  Go  to .ssh file
![1](images/1.png)

## Open an SSH connection to your virtual machine
In your bash prompt, open an SSH connection to your virtual machine. Replace the IP address with the one from your VM, and replace 
the path to the .pem with the path to where the key file was downloaded.

`ssh -i ~/.ssh/file.pem azureuser@ip_address`
  
For easy ssh login, do the following:

* Open a GNU nano editor by entering `nano ~/.ssh/config` in the bash prompt
 
* Setup SSH config file
  
  ```
  Host azure-mlops-zoomcamp
    HostName 20.106.112.123 # VM Public IP
    User azureuser # VM user
    IdentityFile C:\Users\User\.ssh\azurevmkey.pem # Private SSH key file
    StrictHostKeyChecking no
    ServerAliveInterval 60 # To prevent ssh from disconnecting (optional)
    ServerAliveCountMax 525600
  ```

Username gotten from 👇👇👇
<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/pem%20key1.PNG?raw=true" alt="enter username" width="650" height="300"/>
</p> 

Public IP gotten from 👇👇👇
<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/public-ip.PNG?raw=true" width="650" height="300"/>
</p> 
  
Anytime you want to login to ssh instance again, type `ssh azure-mlops-zoomcamp`

### Note: You don't have to rent an instance in the cloud. You can follow the same instructions for setting up your local environment.

## Install Anaconda, Docker and Docker Compose 

### Install Anaconda

* Download and install the Anaconda distribution of Python
  Ensure you are in home directory, if not `cd $HOME`
  
  ```
  wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
  bash Anaconda3-2022.05-Linux-x86_64.sh
  ```

### Install Docker

* Update existing packages using 
  
  `sudo apt update`

* Install Docker using 

  `sudo apt install docker.io`
  


### Install Docker Commpose

* Install docker-compose in a separate directory

  ```
  mkdir soft
  cd soft
  ```
  
* Download Docker Compose
  `wget https://github.com/docker/compose/releases/download/v2.5.0/docker-compose-linux-x86_64 -O docker-compose`
  
  Make it executable

  `chmod +x docker-compose`

  Go back to HOME path
  `cd $HOME` 
  Add to the soft directory to PATH. Open the .bashrc file with nano:

  `nano .bashrc`
  In .bashrc, add the following command to the last line:

  `export PATH="${HOME}/soft:${PATH}"`

  Save it and exit using:  `STRG + X`
  
  Run the following to make sure the changes are applied:

  `source .bashrc`
  
  ## bath must be like that: 
  `(base) azureuser@azvm:~$`
  
## Verify Installation

```
which python
#/home/azureuser/anaconda3/bin/python

which docker
#/usr/bin/docker

which docker-compose
/home/azureuser/soft/docker-compose
```
  
### Run Docker

`docker run hello-world`

If you get

```
docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: 
Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/create": dial unix /var/run/docker.sock: connect: permission denied. error
```

Restart your VM instance in azure's portal or write sudo befor the command

* Run docker without sudo:

  ```
  sudo groupadd docker
  sudo usermod -aG docker $USER
  ```
  

  
## Port Forwarding with Vscode

* Open Vscode
* Install `Remote SSH` extension
* Connecting to Remote SSH host
  
 Go to the bottom left of your Vscode and click the icon that contains 
 mlops-zoomcamp. You might not see any name in yours yet that's because 
 this is your first time.
  
<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/sshvscode.PNG?raw=true" width="700" height="300"/>
</p>

* After clicking, a prompt pops up, Click on `Connect to host` and select your ssh hostname

<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/sshmlops.PNG?raw=true" width="700" height="300"/>
</p>
  
* A new Vscode window opens, Go to View in the menu bar and click on terminal 
  or simply enter `ctrl + ``

* Under Terminal, go to PORTS AND forward port `8888`  
<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/vscode-port.PNG?raw=true" width="600" height="300"/>
</p>

<br />

# Setting up Jupyter Notebook
  
Jupyter notebook is installed with anaconda
  
To run Jupyter Notebook, run the following:

* Create Jupyter notebook Directory

```
cd notebooks
mkdir notebooks
```

* run Jupyter Lab

`jupyter lab`  
 
<p align=center>
<img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/jupyter%20notebook%20init.PNG?raw=true" width="700" height="300"/>
</p>

Take note of the port after localhost(http://127.0.0.1:<port>/

* Jupyter Homepage
  
  <p align=center>
  <img src="https://github.com/josepholaide/MLOps-Practice/blob/main/Week%201/images/notebook%20home.PNG?raw=true" width="900" height="500"/>
  </p>
