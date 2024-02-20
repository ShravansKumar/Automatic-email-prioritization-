!pip install transformers
!pip install tensorflow
!pip install tensorflow-text

!pip freeze > requirements.txt


import tensorflow_hub as hub
import pandas as pd
import tensorflow_text as text
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# loading train test split
from sklearn.model_selection import train_test_split
import tensorflow as tf

#reading dataset
df = pd.read_csv('dataset.csv')
df.head()

#for plotting
sms = pd.value_counts(df["Label"], sort=True)
sms.plot(kind="pie", labels=["ham", "spam"], autopct="%1.0f%%")

plt.title("Email Distribution")
plt.ylabel("")
plt.show()

#to get numeric count
df_spam = df[df['Label']=='spam']
print("Spam Dataset Shape:", df_spam.shape)

df_ham = df[df['Label']=='ham']
print("Ham Dataset Shape:", df_ham.shape)

#downsampling for balancing
df_ham_downsampled = df_ham.sample(df_spam.shape[0])
df_ham_downsampled.shape

#making a new dataset with the downsampled data 
df_balanced = pd.concat([df_spam , df_ham_downsampled])
df_balanced.head()

df_balanced=df_balanced.sample(frac=1)

#adding spam or ham label
df_balanced['spam'] = df_balanced['Label'].apply(lambda x:1 if x=='spam' else 0)

df_balanced['Body'].isna().sum()
df_balanced['Body'] = df_balanced['Body'].fillna('')

#making test and train data
x_train, x_test , y_train, y_test = train_test_split(df_balanced['Body'], df_balanced['spam'],stratify = df_balanced['spam'])
y_train.value_counts()
y_test.value_counts()

#installing bert base model
# downloading preprocessing files and model
bert_preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = 'Inputs')
preprocessed_text = bert_preprocessor(text_input)
embeed = bert_encoder(preprocessed_text)
dropout = tf.keras.layers.Dropout(0.1, name = 'Dropout')(embeed['pooled_output'])
outputs = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'Dense')(dropout)

# creating final model
model = tf.keras.Model(inputs = [text_input], outputs = [outputs])

# check summary of model
model.summary()

Metrics = [tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
           tf.keras.metrics.Precision(name = 'precision'),
           tf.keras.metrics.Recall(name = 'recall')
           ]

model.compile(optimizer ='adam',
               loss = 'binary_crossentropy',
               metrics = Metrics)

#training
history = model.fit(x_train, y_train, epochs = 30 )

# getting y_pred by predicting over X_text and flattening it
y_pred = model.predict(x_test)
y_pred = y_pred.flatten() # require to be in one dimensional array , for easy maniputation


y_pred = np.where(y_pred>0.5,1,0 )
y_pred

# importing consfusion maxtrix
from sklearn.metrics import confusion_matrix , classification_report

cm = confusion_matrix(y_test,y_pred)
cm

# plotting as graph - importing seaborn
import seaborn as sns
sns.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# printing classification report
print(classification_report(y_test , y_pred))

#testing
predict_text=  [
     'We’d all like to get a $10,000 deposit on our bank accounts out of the blue, but winning a prize—especially if you’ve  and get your refund',
                'Your account is temporarily frozen. Please log in to to secure your account ',
                'Netflix is sending you a refund of $12.99. Please reply with your bank account and routing number to verify',
                'The article was published on 18th August itself',
                'Although we are unable to give you an exact time-frame at the moment, I would request you to stay tuned for any updates.',
                'The image you sent is a UI bug, I can check that your article is marked as regular and is not in the monetization program.'
]

test_results = model.predict(predict_text)
output = np.where(test_results>0.5,'spam', 'ham')
output#printing outputs