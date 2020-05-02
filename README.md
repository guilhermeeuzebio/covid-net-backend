# covid-net-backend

This is a simple project to serve an API to covid-net app.

This project is using COVID-Net, links below:

[Paper](https://arxiv.org/pdf/2003.09871.pdf) 
[Code](https://github.com/lindawangg/COVID-Net)

To run this project, first is necessary to download the pretrained model in the below link:

[Pretrained model](https://drive.google.com/drive/folders/1eNidqMyz3isLjGYN1evzQu--A-JVkzbk)

Clone this project and extract the downloaded pretrained model and put it in models. You must have the following in covid-net-backend/app/models :

```
  - COVIDNet-CXR-Large
    - savedModel
      - variables
        - variables.data-00000-of-00001
        - variables.index
      - saved_model.pb
    - checkpoint
    - model.meta 
    - model-8485.data-00000-of-00001
    - model-8485.index
```

Install dependencies:

```
pip3 install -r requirements.txt
```

Run:

```
python3 app.py
```

Send an image in form-data, with key as file, to the below endpoint:

```
/api/v1/covid-inference
```

