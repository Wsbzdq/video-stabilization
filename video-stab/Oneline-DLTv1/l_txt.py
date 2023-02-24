import os
with open("../Data/test.txt", "w") as f:
    for file in os.listdir("../Data/Train"):
        print(file)
        for i in range(len(os.listdir("../Data/Train/"+file))):
            print(i)
            if i ==0:
                f.write(file+"/"+file+"_"+str(10002+i)+".jpg"+" "+file+"/"+file+"_"+str(10001+i)+".jpg"'\n')
                f.write(file+"/"+file + "_" + str(10003 + i) + ".jpg" + " " + file+"/"+file + "_" + str(10001 + i) + ".jpg"+'\n')

            elif(i==1):
                f.write(file+"/"+file + "_" + str(10000 + i) + ".jpg" + " " + file+"/"+file + "_" + str(10001 + i) + ".jpg" + '\n')
                f.write(file+"/"+file + "_" + str(10002 + i) + ".jpg" + " " + file+"/"+file + "_" + str(10001 + i) + ".jpg" + '\n')
                f.write(file+"/"+file + "_" + str(10003 + i) + ".jpg" + " " + file+"/"+file + "_" + str(10001 + i) + ".jpg" + '\n')

            elif(i==(len((os.listdir("../Data/Train/"+file)))-1)):
                f.write(file+"/"+file + "_" + str(10000 + i) + ".jpg" + " " + file+"/"+file + "_" + str(10001 + i) + ".jpg" + '\n')

            elif(i==(len((os.listdir("../Data/Train/"+file)))-2)):
                f.write(file + "/" + file + "_" + str(10000 + i) + ".jpg" + " " + file + "/" + file + "_" + str(
                    10001 + i) + ".jpg" + '\n')
                f.write(file + "/" + file + "_" + str(9999 + i) + ".jpg" + " " + file + "/" + file + "_" + str(
                    10001 + i) + ".jpg" + '\n')
            else:
                f.write(file + "/" + file + "_" + str(9999 + i) + ".jpg" + " " + file + "/" + file + "_" + str(
                    10001 + i) + ".jpg" + '\n')
                f.write(file + "/" + file + "_" + str(10000 + i) + ".jpg" + " " + file + "/" + file + "_" + str(
                    10001 + i) + ".jpg" + '\n')
                f.write(file + "/" + file + "_" + str(10002 + i) + ".jpg" + " " + file + "/" + file + "_" + str(
                    10001 + i) + ".jpg" + '\n')
                f.write(file + "/" + file + "_" + str(10003 + i) + ".jpg" + " " + file + "/" + file + "_" + str(
                    10001 + i) + ".jpg" + '\n')
f.close()



