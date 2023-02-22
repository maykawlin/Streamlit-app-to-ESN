# FUNCTIONS TO HELP WITH PREPROCESSING

def sliding_window(X, width):
    import numpy as np
    
    x = []
    y = []
    for i in range(len(X) - width - 1):
        x.append(X[i : i + width])
        y.append(X[i + width])
    return np.array(x), np.array(y)



def generate_heat_map(x_variable, y_variable, z_variable):
    import pandas as pd
    import numpy as np
    import seaborn as sns

    # create data frame
    d = {'x_variable': x_variable,
         'y_variable': y_variable, 'z_variable': z_variable}
    df = pd.DataFrame(data=d)

    # make heat map
    heatmap_data = pd.pivot_table(df, values='z_variable', index=[
                                  'y_variable'], columns='x_variable')
    sns.heatmap(heatmap_data, cmap='YlGnBu')


# METRICS -----------------------------------------------------------------------------------------------

# Hits -----------------------------------------------------------------------------------------------------

def angular_coef(predict_values, real_values,show = False):
    import numpy as np
    #predict_values = np.vstack([np.reshape(predict_values,(predict_values.shape[0],1)),np.zeros(1)])
    #real_values = np.vstack([np.reshape(real_values,(real_values.shape[0],1)),np.zeros(1)])
    a = np.zeros([predict_values.shape[0]])
    b = np.zeros([real_values.shape[0]])
    acertos = 0
    for n in range(0,predict_values.shape[0]-1):
        a[n] = predict_values[n+1] - real_values[n]
        b[n] = real_values[n+1] - real_values[n] 
        if (a[n]>0) and (b[n]>0):
            acertos += 1
            if show==True:
                print([n+1],'...','Acertou')
        elif (a[n]<0) and (b[n]<0):
            acertos += 1
            if show==True:
                print([n+1],'...','Acertou')
        elif (a[n]==0) and (b[n]==0):
            acertos += 1
            if show==True:
                print([n+1],'...','Acertou')
        else:
            if show==True:
                print([n+1],'...','Errou')
            pass
        #print('Previsão',[n+1],'=',a[n],'......','Real',[n+1],'=',b[n])
    #print('Acertos =',acertos*100/(predict_values.shape[0]-1),'%')
    return acertos*100/(predict_values.shape[0]-1)

# Run_Echos -----------------------------------------------------------------------------------------------


def run_ESN(points,training_input,training_label,testing_input,testing_label,input_size,output_size):
    import numpy as np
    import sys
    sys.path.insert(1, '../ESN')
    from New_ESN import ESN
    
    percentual_acertos = np.zeros([points.shape[0]],dtype=float)
    for n in range(points.shape[0]):
        acerto = []
        n_reservoir =400
        epoch= 2
        samples = 10
        
        
        for j in range(samples):
            
            prediction = np.zeros(100)
            training = len(training_input)-len(prediction)
            
            esn = ESN(input_scaling = 1, epochs = 2, N_u = input_size, N_y = output_size, N_r = n_reservoir, sparsity = 0.2, alpha = points[n,0], beta = 3, verbose = False, method = 'ridge_regression', TF = False)
            
            # Treinando e prevendo
            for i in range(training):
                esn.train(training_input[i], training_label[i])

            # Prevendo
            for i in range(len(prediction)):
                prediction[i] = esn.predict(testing_input[i+training],testing_label[i+training])
                
            y_true = testing_label
            y_pred = prediction
            acerto.append(angular_coef(y_pred, y_true))

        acerto = np.array(acerto)   
        percentual_acertos[n] = np.mean(acerto)
        #print('sparsity =',set_of_points[n,0],'....','spectral_radius =',set_of_points[n,1],'....','alpha =',set_of_points[n,2],
        #,'....','beta =',set_of_points[n,3]'....', 'Acertos =', percentual_acertos[n,k])
    return percentual_acertos


def run_DeepESN(points,training_input,training_label,testing_input,testing_label,input_size,output_size):
    import numpy as np
    import sys
    sys.path.insert(1, '../Deep ESN')
    from New_DeepESN import deep_echo_state_np
    
    percentual_acertos = np.zeros([points.shape[0]],dtype=float)
    for n in range(points.shape[0]):
        acerto = []
        n_reservoir =400
        epoch= 2
        samples = 10
        
        
        for j in range(samples):
            
            prediction = np.zeros(100)
            training = len(training_input)-len(prediction)
            
            deep_echo = deep_echo_state_np(input_scaling = 1, epochs = epoch, N_u = input_size, N_y = output_size, N_r = n_reservoir, N_l = int(np.ceil(points[n,0])), sparsity = 0.2, beta = 3, verbose = False, method = 'ridge_regression')
            
            # Treinando e prevendo
            for i in range(training):
                deep_echo.train(training_input[i], training_label[i])

            # Prevendo
            for i in range(len(prediction)):
                prediction[i] = deep_echo.predict(testing_input[i+training])
                
            y_true = testing_label
            y_pred = prediction
            acerto.append(angular_coef(y_pred, y_true))

        acerto = np.array(acerto)   
        percentual_acertos[n] = np.mean(acerto)
        #print('sparsity =',set_of_points[n,0],'....','spectral_radius =',set_of_points[n,1],'....','alpha =',set_of_points[n,2],
        #,'....','beta =',set_of_points[n,3]'....', 'Acertos =', percentual_acertos[n,k])
    return percentual_acertos


def profit(predict_values, real_values, opening, dates, capital = 10000, limit_errors = 0, time_wait = 0):
    import numpy as np
    from datetime import datetime
    
    format_str = '%d.%m.%Y'
    from_date = dates[0]
    to_date = dates[-1]
    from_date_obj = datetime.strptime(from_date, format_str)
    to_date_obj = datetime.strptime(to_date, format_str)
    num_months = (to_date_obj.year - from_date_obj.year) * 12 + (to_date_obj.month - from_date_obj.month)

    investimento = {"capital": capital, 'investido': 0,"logs": []}

    a = np.zeros([predict_values.shape[0]])
    b = np.zeros([real_values.shape[0]])
    erros = 0
    time_wait_aux = 0
    ações_aux = 0
    last_opening = opening[0]
    
    for n in range(len(predict_values)-1):

        a[n] = predict_values[n+1] - real_values[n] # coeficiente angular da ESN
        b[n] = real_values[n+1] - real_values[n]    # coeficiente angular dos valores reais

        # comprar
        if (a[n]>0):
            ações = np.floor(investimento['capital']/opening[n+1])
            cotas = np.floor(ações/100)
            ações = cotas*100
            if cotas>=1 and time_wait_aux<=0:
                ações_aux += ações
                investimento['investido'] = investimento['investido'] + opening[n+1]*ações
                investimento['capital'] -= opening[n+1]*ações
                last_opening = opening[n+1]
                investimento['logs'].append({'date': dates[n],
                                             'qty': ações_aux,
                                             'share_value': opening[n+1],
                                             'value': investimento['investido'],
                                             'nature': 'buy',
                                             'capital': investimento['capital'],
                                             'score': int((a[n]>0)==(b[n]>0))})
                
                if (b[n]<0):
                    erros += 1
                    if erros >= limit_errors:
                        time_wait_aux = time_wait

            else:
                erros = 0
                time_wait_aux -= 1
                investimento['investido'] = opening[n+1]*ações_aux
                investimento['logs'].append({'date': dates[n],
                                             'qty': ações_aux,
                                             'share_value': opening[n+1],
                                             'value': investimento['investido'],
                                             'nature': 'sem operação',
                                             'capital': investimento['capital'],
                                             'score': int((a[n]>0)==(b[n]>0))})

        # vender
        elif (a[n]<0):
            if ações_aux >=1 and time_wait_aux<=0:
                investimento['capital'] = investimento['capital'] + opening[n+1]*ações_aux
                investimento['investido'] = opening[n+1]*ações_aux
                investimento['logs'].append({'date': dates[n],
                                             'qty': ações_aux,
                                             'share_value': opening[n+1],
                                             'value': investimento['investido'],
                                             'nature': 'sell',
                                             'capital': investimento['capital'],
                                             'score': int((a[n]<0)==(b[n]<0))})
                ações_aux = 0
                investimento['investido'] = 0
                
                if (b[n]>0):
                        erros += 1
                        if erros >= limit_errors:
                            time_wait_aux = time_wait
            else:
                erros = 0
                time_wait_aux -= 1
                investimento['logs'].append({'date': dates[n],
                                             'qty': ações_aux,
                                             'share_value': opening[n+1],
                                             'value': investimento['investido'],
                                             'nature': 'sem operação',
                                             'capital': investimento['capital'],
                                             'score': int((a[n]<0)==(b[n]<0))})
            
        else:
            pass
    
    profit = ((investimento['capital']+investimento['investido'])*100/capital)-100
    month_profit = profit/num_months
    

    return month_profit

def split_data(predictions,raw_data,target,PCA = False,if_pca_dimension=1):
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    
    if PCA==True:
        import sys
        sys.path.insert(1, '../PCA')
        from PCA import PCA
        pca = PCA(n_features=np.array(raw_data[:,:]).shape[1])  
    pass 

    # Number of predictions
    futuretotal = predictions

    trainlen_input = raw_data.shape[0]-futuretotal-1
    trainlen_label = raw_data.shape[0]-futuretotal

    # Train_input
    train_input = []
    for i in range(0,futuretotal):
        train_input.append(np.zeros((trainlen_input+i,raw_data.shape[1])))
    train_input = np.array(train_input)
    for j in range(0,futuretotal):
        train_input[j] = raw_data[:trainlen_input+j,:]

    # Train_label
    train_label = []
    for i in range(0,futuretotal):
        train_label.append(np.zeros((trainlen_label+i,1)))
    train_label = np.squeeze(train_label)
    for j in range(0,futuretotal):
        train_label[j] = np.squeeze(raw_data[1:trainlen_label+j,target])

    # Test_input
    test_input = np.zeros((futuretotal,raw_data.shape[1]))
    for i in range(0,futuretotal):
        test_input[i] = raw_data[trainlen_input+i:trainlen_input+i+1,:]   

    # Test_label
    test_label = np.zeros((futuretotal,1))
    for i in range(0,futuretotal):
        test_label[i] = raw_data[trainlen_label+i:trainlen_label+i+1,target]   
    test_label = np.squeeze(test_label)
    
    
    #usando PCA
    if PCA==True:    
        #Train_input
        lista_train_input = []
        for i in range(0,train_input.shape[0]):
             lista_train_input.append(np.zeros((train_input[i].shape[0],if_pca_dimension)))
        train_input_pca_ = np.array(lista_train_input)
        for i in range(0,train_input.shape[0]):
            train_input_pca_[i] = pca.PCA(train_input[i],if_pca_dimension)

         #Test_input
        lista_test_input = []
        for i in range(0,futuretotal):
            lista_test_input.append(np.zeros((trainlen_input+i,if_pca_dimension)))
        raw_test_input = np.array(lista_test_input)
        for j in range(0,futuretotal):
            raw_test_input[j] = raw_data[:trainlen_input+j+1,:]


        test_input_pca_ = np.zeros((futuretotal,if_pca_dimension))
        for i in range(0,test_input.shape[0]):
            test_input_pca_[i] = pca.PCA(raw_test_input[i],if_pca_dimension)[-1,:]
        
        return train_input,train_label,test_input,test_label,train_input_pca_,test_input_pca_
    
    else:
        return train_input,train_label,test_input,test_label

def split_data_second_form(predictions,input_data,target_data):
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    import sys
    sys.path.insert(1, '../PCA')
    
    # Number of predictions
    futuretotal = predictions

    trainlen_input = input_data.shape[0]-futuretotal-1
    trainlen_label = target_data.shape[0]-futuretotal

    # Train_input
    train_input = []
    for i in range(0,futuretotal):
        train_input.append(np.zeros((trainlen_input+i,input_data.shape[1])))
    train_input = np.array(train_input)
    for j in range(0,futuretotal):
        train_input[j] = input_data[:trainlen_input+j,:]

    # Train_label
    train_label = []
    for i in range(0,futuretotal):
        train_label.append(np.zeros((trainlen_label+i,1)))
    train_label = np.array(train_label)
    for j in range(0,futuretotal):
        train_label[j] = target_data[1:trainlen_label+j,:]

    # Test_input
    test_input = np.zeros((futuretotal,input_data.shape[1]))
    for i in range(0,futuretotal):
        test_input[i] = input_data[trainlen_input+i:trainlen_input+i+1,:]   

    # Test_label
    test_label = np.zeros((futuretotal,1))
    for i in range(0,futuretotal):
        test_label[i] = target_data[trainlen_label+i:trainlen_label+i+1,:]   
    test_label = np.squeeze(test_label)
    return train_input,train_label,test_input,test_label


def get_stocks_from_list(dataframe_names):    
    # Libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import investpy as inv
    from datetime import date
    today = date.today().strftime("%d/%m/%Y")

    # Names of Stocks/Index/ Commodities
    #data_names = pd.read_csv('../Dados/Investing.com/Processed/data_names.csv').drop(['Unnamed: 0'],axis=1)
    data_names = dataframe_names
    data_names = data_names.sort_values(['country']).reset_index(drop=True)
    data_names = data_names.fillna('null', inplace=False)

    raw_data = []
    for i in np.array(data_names):
        if i[6] != 'null':
            raw_data.append(inv.get_index_historical_data(i[1],country = i[0],from_date='01/01/2000',to_date=today))
        elif i[12] != 'null':
            raw_data.append(inv.get_commodity_historical_data(i[1],from_date='01/01/2000',to_date=today))
        else:
            try:
                raw_data.append(inv.get_stock_historical_data(i[5],country = i[0],from_date='01/01/2000',to_date=today))
            except:
                raw_data.append(inv.get_currency_cross_historical_data(i[1],from_date='01/01/2000',to_date=today))

    # Creating a Date column
    data = raw_data
    for i in range(0,len(data)):
        date = data[i].index
        data[i]['Date'] = date
        data[i] = data[i].reset_index(drop=True)

    # Creating the reference
    df1 = data[0]
    for i in data:
        df2 = i
        a = df1['Date'].isin(df2['Date'])
        b = np.array(pd.DataFrame(np.where(a==False)))[0]
        for j in range(b.shape[0]):
            df1 = df1.drop(b[j],axis=0)
        df1 = df1.reset_index(drop=True)
    reference = df1.copy()
    reference_date = df1['Date'].copy()

    # Adjusting all the datas to the reference
    df = []
    for i in data:
        df2 = i
        a = df2['Date'].isin(reference['Date'])
        b = np.array(pd.DataFrame(np.where(a==False)))[0]
        for k in range(b.shape[0]):
            df2 = df2.drop(b[k])
        df2 = df2.reset_index(drop=True)
        df.append(df2)

    # Putting all the datas together
    data_frame = df[0]
    for l in range(1,len(df)):
        data_frame = pd.concat([data_frame,df[l]],axis=1)
    data_frame = data_frame.drop(['Volume','Currency','Date'],axis=1)
    data_frame = pd.concat([reference_date,data_frame],axis=1)
    names = np.array(data_names['name'])

    # Creating columns names
    aux = 0
    columns = []
    teste = data_frame.copy()
    for j in range(1,np.array(teste).shape[1],4):
        columns.append([str(j)+'.'+'Open'+'_'+names[aux],str(j+1)+'.'+'High'+'_'+names[aux],
                        str(j+2)+'.'+'Low'+'_'+names[aux],str(j+3)+'.'+'Close'+'_'+names[aux]]) 
        aux +=1

    columns = np.reshape(np.array(columns),(1,len(columns)*4))[:]

    # trasforming columns array in list
    list_names = []
    list_names.append('Date')
    for i in range(0,len(columns[0])):
        list_names.append(columns[0][i])
    columns = list_names

    # Setting columns names on data frame
    data_frame = data_frame.set_axis(columns, axis=1, inplace=False)
    
    return data_frame
