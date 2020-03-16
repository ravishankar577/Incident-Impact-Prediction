from flask import Flask, request, render_template
import pandas as pd
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from flask import send_file
#from imblearn.ensemble import BalancedBaggingClassifier


app = Flask(__name__,template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        sub_inp=df[['ID', 'ID_status', 'count_reassign', 'count_updated', 'opened_time',
           'updated_by', 'updated_at', 'confirmation_check', 'ID_caller',
           'opened_by', 'location', 'category_ID', 'user_symptom', 'Support_group',
           'support_incharge']]
 
        label_enc_sub_inp_dict = {}
        sub_inp_enc=sub_inp.copy()
        # Loop over columns to encode
        for col_name in sub_inp_enc:
            # Create ordinal encoder for the column
            label_enc_sub_inp_dict[col_name] = LabelEncoder()
            # Select the non-null values in the column
            col = sub_inp_enc[col_name]
            col_not_null = col[col.notnull()]
            reshaped_vals = col_not_null.values.reshape(-1, 1)
            #Encode the non-null values of the column
            encoded_vals = label_enc_sub_inp_dict[col_name].fit_transform(reshaped_vals)
            #Replace the columns with label encoded values
            sub_inp_enc.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
        
        model = pickle.load(open("Ensemble.pkl", "rb"))
        result1= model.predict(sub_inp_enc) 
        ss1 = pd.DataFrame(result1, columns=[ 'prediction'])
        ss1.loc[ss1['prediction']== 1,'prediction']="2 - Medium"
        ss1.loc[ss1['prediction']== 0,'prediction']="1 - High"
        ss1.loc[ss1['prediction']== 2,'prediction']="3 - Low"
        output = pd.concat([df, ss1],axis=1)
        output.to_csv(r'output.csv',index=False)
        
        return render_template('upload.html', result=ss1)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)