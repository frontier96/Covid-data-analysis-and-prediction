# mortality_rate = data_states1['Mortality_Rate']
# #operate_data_counties1['mortality_rate'] = operate_data_counties1[operate_data_counties1['State'].isin(mortality_rate.index)]

# plt.figure(figsize = (5,5))
# sns.distplot(data_states1['Mortality_Rate'])
# plt.figure(figsize = (5,5))
# sns.scatterplot(x=data_states1['Province_State'],y=data_states1['Mortality_Rate'])





#added 5/10 sunday by Jack
mortality_rate = data_states1['Mortality_Rate']
#operate_data_counties1['mortality_rate'] = operate_data_counties1[operate_data_counties1['State'].isin(mortality_rate.index)]

plt.figure(figsize = (10,3))
sns.distplot(data_states1['Mortality_Rate'])
plt.figure(figsize = (10,7))
p1 = sns.scatterplot(x=data_states1['People_Tested'],y=data_states1['Mortality_Rate'])

for line in range(0,data_states1.shape[0]):
  if data_states1['People_Tested'][line] >0 and data_states1['Mortality_Rate'][line] >0:
     p1.text(data_states1['People_Tested'][line]+0.2, data_states1['Mortality_Rate'][line], data_states1['Province_State'][line],
         #data_states1['People_Tested'],data_states1['Mortality_Rate'],data_states1['Province_State'], 
             horizontalalignment='left', size='small', color='black') #, weight='semibold')





#mortality_rate.value_counts()
#data_states1[data_states1['Mortality_Rate']>8]





state_abbreviation = data_counties1.groupby(['StateName', 'State']).agg(sum)#['StateName']
state_dict_original = state_abbreviation.reset_index()[['StateName', 'State']]
new_state_df = pd.DataFrame({'StateName': ['AK','VI', 'PR', "HI", 'GU',"AS",'MP'], 
                             'State': ['Alaska', 'Virgin Islands','Puerto Rico','Hawaii', "Guam",'American Samoa'
                                      ,'Northern Marianas']})
state_dict_combined = pd.concat([state_dict_original, new_state_df])
state_dict = state_dict_combined.set_index('StateName')['State']
mapped_state = data_counties1['StateName'].map(state_dict)
mapped_state.isna().sum()
#state_dict




data_counties1['State updated'] = mapped_state
mortality_dict = data_states1[data_states1['Country_Region'] == 'US'][['Province_State', 'Mortality_Rate']].set_index('Province_State')['Mortality_Rate']
mapped_mortality = data_counties1['State updated'].map(mortality_dict)
mapped_mortality.isna().sum() #should be American Samoa 
data_counties1['Mortality Rate'] = mapped_mortality
data_counties1.head()





first_case_count = data_conf1['First_Case'].value_counts()
plt.figure(figsize=(12, 9))
plt.bar(first_case_count.index, first_case_count.values)
plt.title("Distribution of First discovered case by county")




first_death_count = data_death1['First_Death'].value_counts()
plt.figure(figsize=(12, 9))
plt.bar(first_death_count.index, first_death_count.values)
plt.show()




operate_data_counties1 = data_counties1[['CountyName', 'State updated', 'stay at home', 'public schools', '>500 gatherings', 'entertainment/gym', 'restaurant dine-in',
                                 'Mortality Rate']][data_counties1['State updated'] != 'American Samoa'][data_counties1['State updated'] != 'Northern Marianas']
#operate_data_counties1[operate_data_counties1['CountyName']=='Washington']
operate_data_counties1['Mortality Rate'] = operate_data_counties1['Mortality Rate'].fillna(3.413353)
operate_data_counties1_with_states = operate_data_counties1
operate_data_counties1_with_states




operate_data_counties1.isnull().sum()




operate_data_counties1_with_states['stay at home'] = operate_data_counties1_with_states['stay at home'].fillna(np.mean(operate_data_counties1_with_states['stay at home']))
operate_data_counties1_with_states['public schools'] = operate_data_counties1_with_states['public schools'].fillna(np.mean(operate_data_counties1_with_states['public schools']))
operate_data_counties1_with_states['>500 gatherings'] = operate_data_counties1_with_states['>500 gatherings'].fillna(np.mean(operate_data_counties1_with_states['>500 gatherings']))
operate_data_counties1_with_states.isnull().sum()
#operate_data_counties1_with_states[operate_data_counties1_with_states['stay at home'].isnull()]





#operate_data_counties1_with_states['stay at home'].value_counts()
#operate_data_counties1_with_states['public schools'].value_counts()
#operate_data_counties1_with_states['>500 gatherings'].value_counts()




operate_data_counties1_with_states = operate_data_counties1_with_states.merge(data_conf1[["", "First_Case", "First_Hundred_Case"]], on = )






data_counties1_PCA = data_counties1.select_dtypes(['number']).drop(columns=['STATEFP','COUNTYFP'])
# center our data and normalize the variance
df_mean = np.mean(data_counties1_PCA)
df_centered = data_counties1_PCA - df_mean
df_centered_scaled = df_centered / (np.var(df_centered))**0.5
data_counties1_PCA = df_centered_scaled
data_counties1_PCA_fillna =data_counties1_PCA.fillna(method = 'ffill') #use the previous valid data to fill NaN,
                                                    #good here since closeby county likely to be in the same State

data_counties1_PCA_fillna2 = data_counties1_PCA_fillna.fillna(0) #fill NaN with no previous valid data (whole column is NaN)
#sum(data_counties1_PCA_fillna2.isna().sum())
data_counties1_PCA_fillna2




#PCA 
u, s, vt = np.linalg.svd(data_counties1_PCA_fillna2, full_matrices=False)
P = u @ np.diag(s)
df_1st_2_pcs =pd.DataFrame(P[:,0:2], columns=['pc1', 'pc2'])
first_2_pcs = df_1st_2_pcs

#jittered scatter plot (added noise)
first_2_pcs_jittered = first_2_pcs + np.random.normal(0, 0.1, size = (len(first_2_pcs), 2))
sns.scatterplot(data = first_2_pcs_jittered, x = "pc1", y = "pc2");

#a better looking scatter plot with labels
#import plotly.express as px
#px.scatter(data_frame = first_2_pcs_jittered, x = "pc1", y = "pc2", text = list(df_1972_to_2016.index)).update_traces(textposition = 'top center')





#scree plot
plt.figure(figsize = (10,10))
x = list(range(1, s.shape[0]+1)) 
plt.plot(x, s**2 / sum(s**2)); 
plt.xticks(x, x);
plt.xlabel('PC #');
plt.ylabel('Fraction of Variance Explained');






from sklearn.model_selection import train_test_split

train, test = train_test_split(operate_data_counties1_with_states, test_size=0.1, random_state=42)






plt.figure(figsize = (5,5))
#sns.regplot(operate_data_counties1_with_states['Mortality_Rate'])




operate_data_counties1_with_states = operate_data_counties1_with_states.merge(data_first_case, how = "inner", left_on = ['CountyName', 'State updated'], right_on = ['County_Name', 'Province_State'])
operate_data_counties1_with_states = operate_data_counties1_with_states.merge(data_first_death, how = "left", left_on = ['CountyName', 'State updated'], right_on = ['County_Name', 'Province_State'])

operate_data_counties1_with_states.head()





operate_data_counties1_with_states.drop(['Province_State_x', 'Province_State_y', 'County_Name_y', 'County_Name_x'], axis = 1, inplace = True)
operate_data_counties1_with_states.head()







time_since_first_case = operate_data_counties1_with_states.copy()
time_since_first_case['stay at home'] = time_since_first_case['stay at home'] - time_since_first_case['First_Case']
time_since_first_case['public schools'] = time_since_first_case['public schools'] - time_since_first_case['First_Case']
time_since_first_case['entertainment/gym'] = time_since_first_case['entertainment/gym'] - time_since_first_case['First_Case']
time_since_first_case['>500 gatherings'] = time_since_first_case['>500 gatherings'] - time_since_first_case['First_Case']
time_since_first_case['restaurant dine-in'] = time_since_first_case['restaurant dine-in'] - time_since_first_case['First_Case']
time_since_first_case.head()









X_train = train.drop(['CountyName', 'State updated','Mortality Rate'], axis=1)
Y_train = train['Mortality Rate']

X_train[:5], Y_train[:5]









from sklearn.linear_model import LinearRegression
from sklearn import metrics



model = LinearRegression(fit_intercept=True) # should fit intercept be true?
model.fit(X_train, Y_train)

Y_prediction = model.predict(X_train)


training_loss = metrics.mean_squared_error(Y_prediction, Y_train)
print("Training loss: ", training_loss)








plt.figure(figsize = (5,5))
sns.regplot(Y_prediction, Y_train)









plt.figure(figsize = (5,5))
sns.regplot(Y_prediction, Y_train-Y_prediction)






# perform cross validation
from sklearn import model_selection as ms

# finding which features to use using Cross Validation
errors = []
range_of_num_features = range(1, X_train.shape[1] + 1)
for N in range_of_num_features:
    print(f"Trying first {N} features")
    model = LinearRegression()
    
    # compute the cross validation error
    error = ms.cross_val_score(model, X_train.iloc[:, 0:N], Y_train).mean()
    
    print("\tScore:", error)
    errors.append(error)

best_num_features = np.argmax(errors) + 1
print (best_num_features)
best_err = min(errors)

print(f"Best choice, use the first {best_num_features} features")



#===================================================================================================================























































































































































































































































































































































