import pandas as pd, re

df = pd.read_excel("nutrition.xlsx")
df = df[['name','protein', 'calories', 'calcium', 'total_fat', 'saturated_fat', 'cholesterol', 'sodium',
'carbohydrate', 'fat', 'fiber', 'saturated_fatty_acids', 'monounsaturated_fatty_acids',
'polyunsaturated_fatty_acids', 'fatty_acids_total_trans']]
 
df['name'] = df['name'].str.lower()
 
name_parts = df['name'].str.split(', ', expand=True)
df['food_name'] = name_parts[0]
df['food_type_1'] = name_parts[1]
df['food_type_2'] = name_parts[2]
df = df.drop('name', axis=1)
 
numeric_columns = ['protein', 'calories', 'calcium', 'total_fat', 'saturated_fat', 'cholesterol', 
                   'sodium', 'carbohydrate', 'fat', 'fiber', 'saturated_fatty_acids', 
                   'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids', 
                   'fatty_acids_total_trans']
 
def extract_numeric_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return value
    
    str_value = str(value).strip().lower()
     
    match = re.match(r'^(\d+\.?\d*)\s*([a-z]*)', str_value)
    if match:
        number = float(match.group(1))
        unit = match.group(2)
 
        if unit == 'mg':
            return number / 1000
        else:
            return number
    else:
        return None

for col in numeric_columns:
    df[col] = df[col].apply(extract_numeric_value)
 
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Data types after conversion:")
print(df.dtypes)

'''
protein                        float64
calories                         int64
calcium                        float64
total_fat                      float64
saturated_fat                  float64
cholesterol                    float64
sodium                         float64
carbohydrate                   float64
fat                            float64
fiber                          float64
saturated_fatty_acids          float64
monounsaturated_fatty_acids    float64
polyunsaturated_fatty_acids    float64
fatty_acids_total_trans        float64
food_name                       object
food_type_1                     object
food_type_2                     object
'''





