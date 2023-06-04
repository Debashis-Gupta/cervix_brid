import streamlit as st
import tensorflow as tf
from PIL import Image
from util import classify_image
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle
import json as js
# set title

st.title("Cervix Cancer Detection")
# Define a list of choices
choices = [0,1]
# Create a text input field
age = st.number_input("Enter your age",value=0)
num_sex_part = st.number_input("Number of sexual partners",value=0)
first_sex_inter = st.number_input("First sexual intercourse at age",value=0)
num_preg = st.number_input("Num of pregnancies",value=0)
smoker = st.selectbox("Do you smoke? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
if smoker!=0:
    num_smokes = st.number_input("Enter the number of years",value=0)
    num_smokes_per_year = st.number_input("Enter the number of years")
else:
    num_smokes=0.0
    num_smokes_per_year=0.0
hormonal_contra = st.selectbox("Do you Hormonal Contraceptives? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
if hormonal_contra!=0:
    hormonal_contra_year = st.number_input("Enter the number of Contraceptives years",value=0)
else:
    hormonal_contra_year=0.0


IUD = st.selectbox("Do you IUD? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
if IUD!=0:
    
    IUD_year = st.number_input("Enter the number of IUD years",value=0)
else: 
    IUD_year=0.0

STD = st.selectbox("Do you STD? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
if STD!=0:
    
    STD_year = st.number_input("Enter the number of STD years",value=0)
else: 
    STD_year=0.0
STD_condylomatosis = st.selectbox("Do you STDs condylomatosis? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_cervical_condylomatosis = st.selectbox("Do you STDs cervical condylomatosis? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_vaginal_condylomatosis = st.selectbox("Do you STDs vaginal condylomatosis? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_vulvo_perineal_condylomatosis = st.selectbox("Do you STDs vulvo-perineal condylomatosis? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_syphilis = st.selectbox("Do you STDs syphilis? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_pelvic_inflammatory_disease = st.selectbox("Do you STDs pelvic inflammatory disease? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_genital_herpes = st.selectbox("Do you STD_genital_herpes? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_molluscum_contagiosum = st.selectbox("Do you STDs molluscum contagiosum disease? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_AIDS = st.selectbox("Do you STDs AIDS disease? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_HIV = st.selectbox("Do you STDs HIV disease? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_Hepatitis_B = st.selectbox("Do you STDs Hepatitis B disease? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_HPV = st.selectbox("Do you STDs HPV disease? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
STD_num_diagnosis = st.number_input("Enter the number of STD  Number of diagnosis",value=0)
STDs_Time_first_diagnosis = st.number_input("Enter the number of STDs  Time since first diagnosis",value=0)
STDs_Time_last_diagnosis = st.number_input("Enter the number of STDs  Time since last diagnosis",value=0)
Dx_cin= st.selectbox("Do you Dx_cin ? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
Dx_HPV = st.selectbox("Do you Dx_HPV ? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
Dx = st.selectbox("Do you Dx ? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
Hinselmann = st.selectbox("Do you Hinselmann ? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
Schiller = st.selectbox("Do you Schiller ? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
Citology = st.selectbox("Do you Citology ? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')
Biopsy = st.selectbox("Do you Biopsy ? Please Select '0' for 'NO' OR '1' for 'YES'",choices,label_visibility='visible')

X_feat_dict = {
    'age':age,
    'number_of_sex_partner':num_sex_part,
    'first_sex_intercourse_at_age':first_sex_inter,
    'num_pregnancy':num_preg,
    'Smokes':smoker,
    'num_smokes':num_smokes,
    'num_smokes_per_year':num_smokes_per_year,
    'hormonal_contra':hormonal_contra,
    'hormonal_contra_year':hormonal_contra_year,
    'IUD':IUD,
    'IUD_year':IUD_year,
    'STD':STD,
    'STD_year':STD_year,
    'STD_condylomatosis':STD_condylomatosis,
    'STD_cervical_condylomatosis':STD_cervical_condylomatosis,
    'STD_vaginal_condylomatosis':STD_vaginal_condylomatosis,
    'STD_vulvo_perineal_condylomatosis':STD_vulvo_perineal_condylomatosis,
    'STD_syphilis':STD_syphilis,
    'STD_pelvic_inflammatory_disease':STD_pelvic_inflammatory_disease,
    'STD_genital_herpes':STD_genital_herpes,
    'STD_molluscum_contagiosum':STD_molluscum_contagiosum,
    'STD_AIDS':STD_AIDS,
    'STD_HIV':STD_HIV,
    'STD_Hepatitis_B':STD_Hepatitis_B,
    'STD_HPV':STD_HPV,
    'STD_num_diagnosis':STD_num_diagnosis,
    'STDs_Time_first_diagnosis':STDs_Time_first_diagnosis,
    'STDs_Time_last_diagnosis':STDs_Time_last_diagnosis,
    'Dx_cin':Dx_cin,
    'Dx_HPV':Dx_HPV,
    'Dx':Dx,
    'Hinselmann':Hinselmann,
    'Schiller':Schiller,
    'Citology':Citology,
    'Biopsy':Biopsy
}
X_feat_dict_list = list(X_feat_dict.values())
print(f"Length of list: {len(X_feat_dict_list)}")
X_feat_np = np.array(X_feat_dict_list,dtype=np.float32)
# X_feat_dict_list_key = list(X_feat_dict.keys())
# print(X_feat_dict_list)
print(f'Numpy array : {X_feat_np}')

file_path = 'model/moz_svm.pkl'
with open(file_path,'rb') as file:
    loaded_model = pickle.load(file)
X_feat_np=X_feat_np.reshape(1,-1)
print(f"Shape of X_feat_np : {X_feat_np.shape}")
# pred = loaded_model.predict(X_feat_np)
# print(f"Predition: {pred}")
# model = pickle.load('model/LR.pkl','r')

# print(f"Type of loaded_model : {loaded_model}")

# X_feat = pd.DataFrame(X_feat_dict_list,columns=X_feat_dict_list_key)
# X_feat = pd.DataFrame.from_dict(X_feat_dict)
# print(f"Dict: {X_feat_dict}")
# CSS to inject contained in a string
# hide_dataframe_row_index = """
#             <style>
#             .row_heading.level0 {display:none}
#             .blank {display:none}
#             </style>
#             """
# st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

# pipeline = Pipeline([
#     ("scaler", RobustScaler()),
#     ("pca", PCA(n_components=13))
# ])
# X_train_transform = pipeline.fit_transform(X_feat_np)
# print(f"Shape of X_train_transform : {X_train_transform.shape}")
# st.table(X_train_transform)
# print(f"X_train_transform:{X_train_transform}")
# print(X_train_transform.shape)
trad_y_pred = loaded_model.predict(X_feat_np)
trad_y_pred_probability = loaded_model.predict_proba(X_feat_np)
print(f"Probability : {trad_y_pred_probability}")
print(f"Traditional Prediction: {trad_y_pred}")
# print(f'type of {type(trad_y_pred)}')
if trad_y_pred == 0:
    trad_pred = False
else:
    trad_pred = True
# st.dataframe(X_feat)







# Display the input value
# st.write("Hello", str(user_input))

# set header
st.header("Please upload your cervix image")

# upload image file

file = st.file_uploader('', type=['jpeg','jpg','png'])

# load classifier
model = tf.keras.models.load_model('model/CervixMed.h5')

# load class names

with open('model/labels.txt') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
# print(class_names)

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image,use_column_width=True)

    # classify image
    class_name, conf_score = classify_image(image,model, class_names)

    if class_name == 'Parabasal':
        cv_pred = False
    else:
        cv_pred = True 
    #write classification


    if(trad_pred==True and cv_pred==True):
        final_prediction='Positive'
    elif(trad_pred==False and cv_pred==False):
        final_prediction='Negative'
    elif(trad_pred==True and cv_pred==False):
        final_prediction='Negative'
    else:
        final_prediction='Positive'



    result = {
        'Final Output': final_prediction
    }
    js_data = js.dumps(result)
    st.json(js_data)
    # df= pd.DataFrame(result,columns=result.keys())
    # st.write("## {}".format(class_name))
    # st.write("### score {}".format(conf_score))
    # st.table(df)

