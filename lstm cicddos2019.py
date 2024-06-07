import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import resample
from sklearn import preprocessing

import os
dataset_path = []

for dirname, _, filenames in os.walk('D:/User/диплом ББСО-03-18/lstm/input'):
    for filename in filenames:
        if filename.endswith('.csv'):
            dfp = os.path.join(dirname, filename)
            dataset_path.append(dfp)



cols = list(pd.read_csv(dataset_path[1], nrows=1))

def load_file(path):
    # data = pd.read_csv(path, sep=',')
    data = pd.read_csv(path,
                   usecols =[i for i in cols if i != " Source IP" 
                             and i != ' Destination IP' and i != 'Flow ID' 
                             and i != 'SimillarHTTP' and i != 'Unnamed: 0'])  # 指定某些列

    return data

samples = pd.concat([load_file(dfp) for dfp in dataset_path], ignore_index=True)  #concat是合并多个数据帧（DataFrame）  这是处理不同数据源的数据并将它们统一为一个数据集


import pandas as pd
import matplotlib.pyplot as plt

# Count the occurrences of each label  计算label的类别 
label_counts = samples[' Label'].value_counts()  

# Create a bar plot to visualize the label counts创建条形图以可视化标签计数
plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar')
plt.title('Comparison of Label Column')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()




def string2numeric_hash(text):  # 用于将文本转换为一个基于 MD5 哈希的数值
    import hashlib
    return int(hashlib.md5(text).hexdigest()[:8], 16)  # 传递给 hashlib.md5() 的数据应该是字节类型而非字符串类型
    # chat老师修正：return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16) 
    # 函数现在能够安全地接收一个字符串，将其转换为字节类型，通过 MD5 哈希算法处理，并从生成的哈希值中提取前8个十六进制字符，最后将这些字符转换为一个十进制整数。
    # 作用是：     

# Flows Packet/s e Bytes/s - Replace infinity by 0
samples = samples.replace('Infinity','0')  # 替换所有的 'Infinity' 字符串为 '0
samples = samples.replace(np.inf,0)  # numpy 表示的无限大值（np.inf）替换为 0 
#samples = samples.replace('nan','0')  # 将所有的 'nan'（不是数字的值）替换为 '0' 可以简化分析，但这可能会影响数据的统计特性
samples[' Flow Packets/s'] = pd.to_numeric(samples[' Flow Packets/s'])  # 确保 samples 数据帧中的 ' Flow Packets/s' 列是数值型; pd.to_numeric 函数尝试将列中的每个值转换为数值类型，如果转换失败，默认行为是将无法解析的值设置为 NaN（这个行为可以通过参数调整）。 

samples['Flow Bytes/s'] = samples['Flow Bytes/s'].fillna(0)  # Flow Bytes/s' 列中的所有 NaN 值替换为 0
samples['Flow Bytes/s'] = pd.to_numeric(samples['Flow Bytes/s'])  # 确保 'Flow Bytes/s' 列也是数值型。使用 pd.to_numeric 进行类型转换。
# 作用：处理了数据中的无效或特殊值（如无限大和 NaN），并确保了关键列是数值型


#Label
from sklearn.preprocessing import LabelEncoder
# Create a label encoder object
label_encoder = LabelEncoder()

# Fit the label encoder to the label column and transform the labels将标签编码器适配到标签列并转换标签
samples[' Label'] = label_encoder.fit_transform(samples[' Label'])  # transform 方法将原始的类别标签转换为这些整数值；结果是一个整数数组，其中每个原始的类别标签都被替换为对应的整数

# Get the mapping between original labels and encoded values 获取原始标签和编码值之间的映射
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))  # label_mapping 字典使得你可以快速查找任何类别标签对应的数值，例如，如果你想知道类别 'cat' 被编码成了什么数字，你可以简单地查看 label_mapping['cat']

# Print the mapping 
for label, encoded_value in label_mapping.items():
    print(f"Label: {label} - Encoded Value: {encoded_value}")
    
#Timestamp - Drop day, then convert hour, minute and seconds to hashing  #时间戳-删除日期，然后将小时、分钟和秒转换为哈希
colunaTime = pd.DataFrame(samples[' Timestamp'].str.split(' ',1).tolist(), columns = ['dia','horas'])   # Timestamp' 列中的每个时间戳字符串基于第一个空格分割， 1 代表分割一次；分割结果是两部分：日期和时间
# tolist() 将分割后的结果转换为列表形式  columns = ['dia','horas']是使用这个列表来创建一个新的数据帧 colunaTime，其中包含两列：'dia'（日期）和 'horas'（时间）

colunaTime = pd.DataFrame(colunaTime['horas'].str.split('.',1).tolist(),columns = ['horas','milisec'])  # 处理 colunaTime 中的 'horas' 列，进一步将时间字符串基于第一个点（.）分割，分割成时间和毫秒
# 结果被转换为列表，并用来创建新的 colunaTime 数据帧，现在包含 'horas'（时间）和 'milisec'（毫秒）两列。

stringHoras = pd.DataFrame(colunaTime['horas'].str.encode('utf-8'))  # 将更新后的 colunaTime 数据帧中的 'horas' 列（时间部分，不包括毫秒）转换为 UTF-8 编码的字节字符串
samples[' Timestamp'] = pd.DataFrame(stringHoras['horas'].apply(string2numeric_hash)) #colunaTime['horas']
# 使用前面定义的 string2numeric_hash 函数将 stringHoras 数据帧中的 'horas' 列（现在是字节字符串）转换为数值哈希。
del colunaTime,stringHoras  # 删除了在过程中创建的临时数据帧 colunaTime 和 stringHoras
# 整个过程的目的是将时间戳中的时间部分从字符串格式转换为数值格式

print('Training data processed')


from keras.models import Sequential
from keras.layers import Dense,Embedding,Dropout,Flatten,MaxPooling1D,LSTM


def LSTM_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(LSTM(32,input_shape=(input_size,1), return_sequences=False))
    model.add(Dropout(0.5))    
    model.add(Dense(10, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(18, activation='softmax'))  # Update units and activation function

    print(model.summary())
    
    return model



def train_test(samples):
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Specify the data 
    X=samples.iloc[:,0:(samples.shape[1]-1)]  # samples.shape[1] 获取数据帧的列数，然后减1来获取最后一列的索引前一个位置，即不包括最后一列
    
    # Specify the target labels and flatten the array
    #y= np.ravel(amostras.type)
    y= samples.iloc[:,-1]  # [-1] 指定选择最后一列
    
    # Split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = train_test(samples)


def format_3d(df):  # 目的是将一个二维数据帧（或类似的数据结构，如二维数组）转换为三维数组
    
    X = np.array(df)  # 将输入的数据帧 df 转换为 NumPy 数组 X
    return np.reshape(X, (X.shape[0], X.shape[1], 1))  # 将 X 重塑为三维数组
# X.shape[0] - 这是 X 的行数，即数据集中的样本数；X.shape[1] - 这是 X 的列数，即每个样本的特征数量；
# 1 - 这是新加的第三维，通常用于为数据增加一个额外的单一维度，这在输入数据到某些类型的神经网络（如只接受三维输入的 CNN）时是必需的；
# 函数的最终输出是一个三维数组，其中每个原始数据点都被扩展为一个具有单一通道维度的三维形式。例如，如果原始数据是形状为 (1000, 10) 的二维数组（1000个样本，每个样本10个特征），则转换后的数组形状将是 (1000, 10, 1)

def compile_train(model,X_train,y_train,deep=True, epochs=10):
    
    if(deep==True):
        import matplotlib.pyplot as plt
#         optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.001)
#         model_lstm2.compile(optimizer=optimizer)

#         model.compile(loss='binary_crossentropy',
#                       optimizer='adam',
#                       metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        history = model.fit(X_train, y_train,epochs=epochs, batch_size=512, verbose=1)
        #model.fit(X_train, y_train,epochs=3)

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        print(model.metrics_names)
    
    else:
        model.fit(X_train, y_train) #SVM, LR, GD
    
    print('Model Compiled and Trained')
    return model


 def save_model(model,name):
    from keras.models import model_from_json
    
    arq_json = 'Models/' + name + '.json'
    model_json = model.to_json()
    with open(arq_json,"w") as json_file:
        json_file.write(model_json)
    
    arq_h5 = 'Models/' + name + '.h5'
    model.save_weights(arq_h5)
    print('Model Saved')
    
def load_model(name):
    from keras.models import model_from_json
    
    arq_json = 'Models/' + name + '.json'
    json_file = open(arq_json,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    arq_h5 = 'Models/' + name + '.h5'
    loaded_model.load_weights(arq_h5)
    
    print('Model loaded')
    
    return loaded_model
    






# UPSAMPLE 
    
X_train, X_test, y_train, y_test = train_test(samples)


#junta novamente pra aumentar o numero de normais
X = pd.concat([X_train, y_train], axis=1)

from imblearn.over_sampling import SMOTE

X_train, y_train = SMOTE(random_state=42).fit_resample(X, X[' Label'])
X_train = pd.DataFrame(X_train, columns=X_train.columns)

X_train=X_train.drop(' Label', axis=1)
input_size = (X_train.shape[1], 1)

del X
import pandas as pd
import matplotlib.pyplot as plt

# Count the occurrences of each label
label_counts = y_train.value_counts()

# Create a bar plot to visualize the label counts
plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar')
plt.title('Comparison of Label Column')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()


from keras.utils import to_categorical

# labels = to_categorical(samples[' Label'])
y_train= to_categorical(y_train)
y_test= to_categorical(y_test)


model_lstm2 = LSTM_model(82)
model_lstm2 = compile_train(model_lstm2,format_3d(X_train.astype(np.float32)),y_train.astype(np.float32), epochs=50)



############################################################################################################

!mkdir Models

!touch 'Models/lstm_50e.json'
save_model(model_lstm2,"lstm_50e")

del model_lstm2

!cp -r /kaggle/input/lstm-ddos-model/* /kaggle/working/Models/
!ls -lah Models

model_lstm = load_model('lstm_50e')
# model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


y_pred = model_lstm.predict(format_3d(X_test)) 

y_pred = y_pred.round()

from sklearn.metrics import accuracy_score
accuracy_score(y_test_label, y_pred_label)

from sklearn.metrics import classification_report
classification_report(y_test_label, y_pred_label)

# !apt -qqqq install aria2

# anngap file pcap sudah diextract menjadi csv
def detect_anomaly(csv_path, model):
    # read data
    data = pd.read_csv(csv_path,
               usecols =[i for i in cols if i != ' Destination IP' and i != 'Flow ID' 
                         and i != 'SimillarHTTP' and i != 'Unnamed: 0'])
    source_ip = data[' Source IP']
    data = data.drop(" Source IP", axis=1)
    
    # only for data from dataset
    data = data.drop(' Label', axis=1)
    
    # preprocess
    def string2numeric_hash(text):
        import hashlib
        return int(hashlib.md5(text).hexdigest()[:8], 16)

    # Flows Packet/s e Bytes/s - Replace infinity by 0
    data = data.replace('Infinity','0')
    data = data.replace(np.inf,0)
    #data = data.replace('nan','0')
    data[' Flow Packets/s'] = pd.to_numeric(data[' Flow Packets/s'])

    data['Flow Bytes/s'] = data['Flow Bytes/s'].fillna(0)
    data['Flow Bytes/s'] = pd.to_numeric(data['Flow Bytes/s'])

    #Timestamp - Drop day, then convert hour, minute and seconds to hashing 
    colunaTime = pd.DataFrame(data[' Timestamp'].str.split(' ',1).tolist(), columns = ['dia','horas'])
    colunaTime = pd.DataFrame(colunaTime['horas'].str.split('.',1).tolist(),columns = ['horas','milisec'])
    stringHoras = pd.DataFrame(colunaTime['horas'].str.encode('utf-8'))
    data[' Timestamp'] = pd.DataFrame(stringHoras['horas'].apply(string2numeric_hash))#colunaTime['horas']
    del colunaTime,stringHoras

    predictions = model.predict(data)
    predictions = np.argmax(predictions, axis=1)

    # Create a DataFrame of anomalies
#     anomaly_df = pd.DataFrame()
#     for i in range(len(predictions)):
#         if predictions[i] != "0":
#             anomaly_df = anomaly_df.append({' Source IP': source_ip[i], "Label": predictions[i]}, ignore_index=True)

    anomaly_df = np.array([])
    for i in range(len(predictions)):
        if predictions[i] != 0:
            anomaly_df = np.append(anomaly_df, [source_ip[i], predictions[i]])
    
    # Print the IP address of the unique anomalies
#     print(anomaly_df["IP"].unique())
    anomaly_df = anomaly_df.reshape(-1, 2)
    label_map = {
        '0': "BENIGN",
        '1': "DrDoS_DNS",
        '2': "DrDoS_LDAP",
        '3': "DrDoS_MSSQL",
        '4': "DrDoS_NTP",
        '5': "DrDoS_NetBIOS",
        '6': "DrDoS_SNMP",
        '7': "DrDoS_UDP",
        '8': "LDAP",
        '9': "MSSQL",
        '10': "NetBIOS",
        '11': "Portmap",
        '12': "Syn",
        '13': "TFTP",
        '14': "UDP",
        '15': "UDP-lag",
        '16': "UDPLag",
        '17': "WebDDoS",
    }
    unique_values = np.unique(anomaly_df, axis=0)

    # Print the results in a fancy format.
    print('Source IP | anomaly')
    print('===================')
    for row in unique_values:
        print(f"{row[0]} | {label_map[row[1]]}")
detect_anomaly('D:/User/диплом ББСО-03-18/lstm/input/cicddos2019/DNS-testing.csv', model_lstm)
