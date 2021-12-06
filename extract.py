import pandas as pd

#Lecture des données txt
with open('attribs.txt', 'r') as f:
    data = [line.strip() for line in f]

#Epuration des données : on garde seulement les data
data2 = []
for i in range(1,len(data)):
    if str(data[i])[:4] == 'data':
        data2.append(str(data[i])[6:])

textfile = open("data.json", "w")

#Ecriture en Json
for element in data2:

    textfile.write(element + "\n")

textfile.close()

#Load JSON + CSV
df = pd.read_json('data.json', lines=True)
print(df)

rssi = df['rssi']
rssi1 = []
rssi2 = []
for i in range(len(rssi)):
    try:
        rssi1.append(rssi[i][0])
        rssi2.append(rssi[i][1])
    except:
        rssi1.append(0)
        rssi2.append(0)

df["rssi1"]=rssi1
df["rssi2"]=rssi2
df2 = df[['type','timestamp','datestamp', 'txAntennaPort','txExpanderPort', 'rssi1', 'rssi2']].copy()
print(df2)
df2.to_csv('data.csv')