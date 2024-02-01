##################################

#Working with Outliers

###################################

import pandas as pd 

my_df = pd.DataFrame({"input1" : [15,41,44,47,50,53,56,59,99],
                      "input2" : [29,41,44,47,50,53,56,59,66]})


# Box Plot Approach

my_df.plot(kind="box",vert=False)

outlier_columns =["input1","input2"]


for column in outlier_columns:
    
    lower_quartile =my_df[column].quantile(0.25)
    upper_quartile =my_df[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    extended_iqr = iqr * 1.5
    min_border = lower_quartile - extended_iqr
    max_border = upper_quartile + extended_iqr
    
    outliers = my_df[(my_df[column]< min_border) | (my_df[column]> max_border)].index
    print(f"{len(outliers)} Outliers dectected in column {column}")
    
    my_df.drop(outliers,inplace=True)
    
    
    
# Standard Deviation Approach

outlier_columns =["input1","input2"]


for column in outlier_columns:
    
    mean = my_df[column].mean()
    std_dev  = my_df[column].std()
    
    min_border = mean - std_dev * 2
    max_border = mean + std_dev * 2 
    
    
    outliers = my_df[(my_df[column]< min_border) | (my_df[column]> max_border)].index
    print(f"{len(outliers)} Outliers dectected in column {column}")
    
    my_df.drop(outliers,inplace=True)









