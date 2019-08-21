import csv
import numpy as np

#Create a list of cells 
def createCellList():
    cellList=[]
    with open("normalized_matrix_after_magic.csv") as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=',')
        next(csv_reader,None)
        total_line=list(csv_reader)
        for line in total_line:
            cellList.append(line[0])
    return cellList



#Create a dictionary which maps the cell with perturbation 
def readDictionaryFile(cellList):
    perturbationList=[]
    with open("GSM2396861_k562_ccycle_cbc_gbc_dict.csv") as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=',')
        total_line=list(csv_reader)
        dictionary=dict()
        newCellList=[]
        for line in total_line:
            perturbationList.append(line[0])
            newline=line[1].split(',')
            for cell in newline:
                if cell[0]==' ' and cell[1:len(cell)] in cellList:
                    if cell[1:len(cell)] in dictionary:
                        dictionary[cell[1:len(cell)]].append(line[0])
                    else:
                        dictionary[cell[1:len(cell)]]=[line[0]]
                elif cell[0]!=' ' and cell[1:len(cell)] in cellList:
                    if cell in dictionary:
                        dictionary[cell].append(line[0])
                    else:
                        dictionary[cell]=[line[0]]
    return perturbationList, dictionary



#Create covariate matrix which is one of the input to linear regression 
def writeCovariateMatrix(cellList, perturbationList, dictionary):
    s = (len(cellList),len(perturbationList))
    d = np.zeros(s)
    for index in range(0, len(d)):
        for i in range(0, len(d[index])):
            if cellList[index] in dictionary:
                if perturbationList[i] in dictionary[cellList[index]]:
                    d[index][i]=1
    output_file=open('covariate_matrix.csv',"w")
    output_csv=csv.writer(output_file)
    for j in range(0, len(d)):
        output_csv.writerow(d[j])
    output_file.close()

            
cellList=createCellList()
perturbationList, dictionary=readDictionaryFile(cellList)
writeCovariateMatrix(cellList, perturbationList, dictionary)
