import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv
from collections import Counter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np
import seaborn as sns


df=pd.read_csv('noHousehold_noSpouse_noFlags_2digit_floats.csv')

filtered_df = df.dropna(subset=['r2demene', 'r3demene', 'r4demene', 'r5demene','r6demene'])

# Print the IDs from column 1 of the filtered DataFrame
print(len(filtered_df['idauniq'].tolist()))

#prints 121232, 121234, 121236, 121238, 121240, 121244, 121247, 121248, 121251, 121252, 121254, 121255, 121256, 121258, 121260, 121261, 121262, 121263, 121265, 121268, 121269, 121273, 121274, 121275, 121283, 121284, 121287, 121289, 121290, 121293, 121294, 121295, 121296, 121299, 121305, 121306, 121309, 121311, 121313, 121314, 121318, 121321, 121322, 
#121323, 121329, 121332, 121333, 121334, 121335, 121338
'''
corr_matrix = df.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(len(to_drop))

#PRINTS 1544 AFTER 15 MINUTES






corrMatrix = df.corr()
sns.set(rc={'figure.figsize':(3,3)})
sns.heatmap(corrMatrix, annot=True).set_title('Correlation between variables')
plt.show()



columns_with_no_na = df.columns[df.isna().sum() >= (len(df)/2)]

print("Columns with no missing values:")
print(columns_with_no_na)
print(sum(df.duplicated()))
corrMatrix = df.corr()
sns.set(rc={'figure.figsize':(3,3)})
sns.heatmap(corrMatrix, annot=True).set_title('Correlation between variables')
plt.show()
     



value_to_find = 1140.88

for col in df.columns:
    if (df[col] == value_to_find).any():
        print(f"First column name with value {value_to_find}: {col}")
        break







float_cols = df.select_dtypes(include=['float']).columns
df[float_cols] = df[float_cols].round(2)

df.to_csv('noHousehold_noSpouse_noFlags_2digit_floats.csv',index=False)

df = df[[col for col in df.columns if len(col) <= 2 or (col[2] != 'f' and col[-1] != 'f')]]

print("\nModified DataFrame:")
print(df.shape)
df.to_csv('noHousehold_noSpouse_noFlags.csv',index=False)



X = add_constant(df)

vif_values = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif_list = list(zip(X.columns, vif_values))
vif_list_sorted = sorted(vif_list, key=lambda x: x[1], reverse=True)

print("Top 20 features with the highest VIF values:")
for feature, vif in vif_list_sorted[:20]:
    print(f"Feature: {feature}, VIF: {vif}")

print(df[df['r2demene']=='1.0'])
value_counts = df['r2demene'].value_counts(dropna=False)

value_counts.plot(kind='bar')
plt.xlabel('Unique Values')
plt.ylabel('Count')
plt.title('Count of Each Unique Value')
plt.show()





print(df.shape)
print(df.describe())
object_columns = df.select_dtypes(include=['object']).columns

counts = {}

for col in object_columns:
    count = df[col].apply(lambda x: x not in ['0', '1', ' ']).sum()
    counts[col] = count

sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

print("Top 5 columns with the highest counts of values other than 0, 1, or ' ':")
for col, count in sorted_counts[5:25]:
    print(f"{col}: {count}")


def count_whitespace(series):
    return series.apply(lambda x: isinstance(x, str) and x.isspace()).sum()

whitespace_counts = df.apply(count_whitespace)

sorted_columns = whitespace_counts.sort_values(ascending=False)

top_5_columns_to_drop = sorted_columns.head(5).index
print(top_5_columns_to_drop)



csv_file = 'filtered_csv_file_startswith_s.csv'  # Replace with your file path
df = pd.read_csv(csv_file,low_memory=False)
def drop_columns_starting_with_h(csv_file):
    df = pd.read_csv(csv_file,low_memory=False)

    columns_to_drop = [col for col in df.columns if col.lower().startswith('h')]

    filtered_df = df.drop(columns=columns_to_drop)

    return filtered_df


filtered_df = drop_columns_starting_with_h(csv_file)

filtered_csv_file = 'elsa_no_h_no_s_no40missing.csv'
filtered_df.to_csv(filtered_csv_file, index=False)


print(filtered_df.shape)

filtered_df = df[df['rabyear'] >= '1960']

print(filtered_df)

counts, edges, bars = plt.hist(data['r1demene'])
            

plt.bar_label(bars)
plt.show()

counts, edges, bars = plt.hist(data['r2demene'])
            

plt.bar_label(bars)
plt.show()
counts, edges, bars = plt.hist(data['r3demene'])
            

plt.bar_label(bars)
plt.show()
counts, edges, bars = plt.hist(data['r4demene'])
            

plt.bar_label(bars)
plt.show()
counts, edges, bars = plt.hist(data['r5demene'])
            

plt.bar_label(bars)
plt.show()
counts, edges, bars = plt.hist(data['r6demene'])
            

plt.bar_label(bars)
plt.show()
counts, edges, bars = plt.hist(data['r7demene'])
            

plt.bar_label(bars)
plt.show()
counts, edges, bars = plt.hist(data['r8demene'])
            

plt.bar_label(bars)
plt.show()
counts, edges, bars = plt.hist(data['r9demene'])
            

plt.bar_label(bars)
plt.show()

#df=pd.read_csv('filtered_csv_file_startswith_s.csv',low_memory=False)
df = pd.read_csv('filtered_csv_file_startswith_s.csv',low_memory=False)
object_columns = df.select_dtypes(include=['object']).columns
df[object_columns] = df[object_columns].apply(pd.to_numeric, errors='coerce')#df = data.apply(pd.to_numeric, errors='coerce')
df.to_csv('numerical_elsa_no_Spouse_no_missing_above_40.csv',index=False)

def count_negative_12(series):
    return (series == '-6100').sum()

negative_12_counts = df.apply(count_negative_12)

sorted_columns = negative_12_counts.sort_values(ascending=False)

top_5_columns_to_drop = sorted_columns.head(5).index
print(top_5_columns_to_drop)

def count_unique_values(csv_file, column_name):
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        values = [row[column_name] for row in reader]
    
    counter = Counter(values)
    
    for value, count in counter.items():
        print(f"Value '{value}' - {count} times")

csv_file = 'numerical_elsa_no_Spouse_no_missing_above_40.csv'
column_name = 'h2aftotf'

count_unique_values(csv_file, column_name)

print(df.shape)
def count_whitespace(series):
    return series.apply(lambda x: isinstance(x, str) and x.isspace()).sum()

whitespace_counts = df.apply(count_whitespace)

sorted_columns = whitespace_counts.sort_values(ascending=False)

print(sorted_columns.head(15))






print(df.shape)
columns_to_remove_startswith_s = [col for col in df.columns if col.lower().startswith('s')]
data_filtered_startswith_s = df.drop(columns=columns_to_remove_startswith_s)

print(f"Columns removed due to starting with 's': {columns_to_remove_startswith_s}")
print(data_filtered_startswith_s.shape)
filtered_file_path_startswith_s = 'filtered_csv_file_startswith_s.csv'
data_filtered_startswith_s.to_csv(filtered_file_path_startswith_s, index=False)


empty_string_percentages = (df == ' ').sum() / len(df) * 100

columns_to_remove = empty_string_percentages[empty_string_percentages > 50].index

data_filtered = df.drop(columns=columns_to_remove)
print(data_filtered.shape)
print(f"Columns removed due to exceeding 40% empty strings: {columns_to_remove}")

# Optionally, save the filtered DataFrame back to a CSV file
filtered_file_path = 'delete50percent_missing.csv'
data_filtered.to_csv(filtered_file_path, index=False)


empty_string_counts = df.apply(lambda x: (x == ' ').sum())

empty_string_counts_sorted = empty_string_counts.sort_values(ascending=False)

print("Columns with the most empty string values:")
print(empty_string_counts_sorted.head(10))


dementia_columns = [col for col in df.columns if 'demene' in col]

print(dementia_columns)


object_columns = df.select_dtypes(include=['object']).columns

selected_object_columns = object_columns[100:120]

selected_df = df[selected_object_columns]

print(selected_df.head(10))


folder_path = 'C:\\Users\\sothr\\Desktop\\research'

unique_ids = set()

for filename in os.listdir(folder_path):
    if filename.endswith('.sav'):
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file
        df = pd.read_spss(file_path)
        # Extract the 'id' column and add the unique IDs to the set
        unique_ids.update(df['idauniq'].tolist())

unique_ids_df = pd.DataFrame(list(unique_ids), columns=['idauniq'])

output_file = 'unique_ids.csv'
unique_ids_df.to_csv(output_file, index=False)

print(f'Unique IDs have been saved to {output_file}')

print(df['r1backward_s'])
missing_percentage = df.isnull().mean() * 100

total_missing_percentage = df.isnull().mean().mean() * 100

print("Percentage of missing values for each column:")
print(missing_percentage)

print("\nTotal percentage of missing values in the DataFrame:")
print(total_missing_percentage)
sorted_missing_percentage = missing_percentage.sort_values(ascending=False)

print("Columns sorted by percentage of missing values (descending):")
print(sorted_missing_percentage)
dtypes = df.dtypes

dtype_counts = dtypes.value_counts()

print(dtype_counts)


missing_val_count_by_column = (df.isnull().sum())
total_cells_by_column = df.shape[0]

pct_missing = (missing_val_count_by_column/total_cells_by_column) * 100
print(pct_missing)
'''