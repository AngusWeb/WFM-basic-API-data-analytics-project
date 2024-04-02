import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


# Load the data from the JSON file
with open("vectis_rivens_data.json", "r") as file:
    data = json.load(file)

# Access the riven auctions from the data
auctions = data["payload"]["auctions"]

# Create a list to store the extracted data
extracted_data = []

# Process each auction and extract relevant information
for auction in auctions:
    item = auction["item"]
    attributes = item["attributes"]
    
    # Create a dictionary for each auction with default values for attributes
    auction_data = {
        "starting_price": auction["starting_price"],
        "buyout_price": auction["buyout_price"],
        "mastery_level": item["mastery_level"],
        "re_rolls": item["re_rolls"],
        "weapon_url_name": item["weapon_url_name"],
        "name": item["name"],
        "polarity": item["polarity"],
        "mod_rank": item["mod_rank"],
        "owner_reputation": auction["owner"]["reputation"],
        "owner_ingame_name": auction["owner"]["ingame_name"],
        "owner_status": auction["owner"]["status"],
        "platform": auction["platform"],
        "created": auction["created"],
        "updated": auction["updated"],
        "attribute_1": "",
        "1:value": float("nan"),
        "attribute_2": "",
        "2:value": float("nan"),
        "attribute_3": "",
        "3:value": float("nan"),
        "negative_attribute": "",
        "value": float("nan")
    }
    
    # Initialize counters for positive and negative attributes
    positive_count = 1
    negative_count = 1
    
    # Iterate over the attributes and assign them to the appropriate columns
    for attr in attributes:
        attr_name = attr["url_name"]
        attr_value = attr["value"]
        
        if attr["positive"]:
            if positive_count <= 3:
                auction_data[f"attribute_{positive_count}"] = attr_name
                auction_data[f"{positive_count}:value"] = attr_value
                positive_count += 1
        else:
            if negative_count == 1:
                auction_data["negative_attribute"] = attr_name
                auction_data["value"] = attr_value
                negative_count += 1
    
    extracted_data.append(auction_data)

# Create a DataFrame from the extracted data
vectisdf = pd.DataFrame(extracted_data)

# Reorder the columns
column_order = [
    "starting_price", "buyout_price", "mastery_level", "re_rolls", "weapon_url_name",
    "name", "polarity", "mod_rank", "owner_reputation", "owner_ingame_name",
    "owner_status", "platform", "created", "updated",
    "attribute_1", "1:value",
    "attribute_2", "2:value",
    "attribute_3", "3:value",
    "negative_attribute", "value"
]
vectisdf = vectisdf[column_order]



# Print the DataFrame
#print(vectisdf.head())



#print(vectisdf.describe())
#print(vectisdf.info())

# Select specific columns and view the top 10 values



attribute_mapping = {
    "base_damage_/_melee_damage": "Damage",
    'toxin_damage': 'Toxin Damage',
    'critical_chance': 'Critical Chance',
    'damage_vs_infested': 'Dmg vs Infested',
    'multishot': 'Multishot',
    'damage_vs_corpus': 'Dmg vs Corpus',
    'electric_damage': 'Electric Damage',
    'impact_damage': 'Impact Damage',
    'slash_damage': 'Slash Damage',
    'damage_vs_grineer': 'Dmg vs Grineer',
    'status_chance': 'Status Chance',
    'Damage': 'Damage',
    'heat_damage': 'Heat Damage',
    'status_duration': 'Status Duration',
    'ammo_maximum': 'Maximum Ammo',
    'reload_speed': 'Reload Speed',
    'recoil': 'Recoil',
    'fire_rate_/_attack_speed': 'Fire Rate',
    'punch_through': 'Punch Through',
    'zoom': 'Zoom',
    'cold_damage': 'Cold Damage',
    'critical_damage': 'Critical Damage',
    'magazine_capacity': 'Magazine Capacity',
    'puncture_damage': 'Puncture Damage'
}

# Create a new DataFrame with only the relevant columns
attribute_price_df = vectisdf[['buyout_price', 'attribute_1', 'attribute_2', 'attribute_3', 'negative_attribute']]

top_10_values = vectisdf.iloc[:10, [0, 1, 3, 5, 6, 14, 15, 16, 17, 18, 19, 20, 21]]

# Print the top 10 values with selected columns
print(top_10_values)


# Melt the DataFrame to combine attribute columns into a single column
melted_df = pd.melt(attribute_price_df, id_vars=['buyout_price'], value_vars=['attribute_1', 'attribute_2', 'attribute_3'], var_name='attribute', value_name='attribute_value')

# Filter out empty attribute values
melted_df = melted_df[melted_df['attribute_value'] != '']

# Map the attribute names using the attribute_mapping dictionary
melted_df['attribute_value'] = melted_df['attribute_value'].map(attribute_mapping)
attribute_price_df['negative_attribute'] = attribute_price_df['negative_attribute'].map(attribute_mapping)

# Calculate the average buyout price for each positive attribute
positive_attribute_impact = melted_df.groupby('attribute_value')['buyout_price'].mean()

# Calculate the average buyout price for each negative attribute
negative_attribute_impact = attribute_price_df.groupby('negative_attribute')['buyout_price'].mean()

# Create a table for positive attributes and sort it by average buyout price in descending order
positive_table = positive_attribute_impact.reset_index()
positive_table.columns = ['Positive Attribute', 'Average Buyout Price']
positive_table = positive_table.sort_values('Average Buyout Price', ascending=False)

# Create a table for negative attributes and sort it by average buyout price in descending order
negative_table = negative_attribute_impact.reset_index()
negative_table.columns = ['Negative Attribute', 'Average Buyout Price']
negative_table = negative_table.sort_values('Average Buyout Price', ascending=False)

# Print the tables
print("Positive Attribute Impact on Buyout Price:")
print(positive_table)
print("\nNegative Attribute Impact on Buyout Price:")
print(negative_table)

# Create a bar graph for positive attributes
plt.figure(figsize=(10, 6))
plt.bar(positive_table['Positive Attribute'], positive_table['Average Buyout Price'])
plt.xlabel('Positive Attribute')
plt.ylabel('Average Buyout Price')
plt.title('Impact of Positive Attributes on Buyout Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('positive_attributes_impact.png')
plt.show()

# Create a bar graph for negative attributes
plt.figure(figsize=(10, 6))
plt.bar(negative_table['Negative Attribute'], negative_table['Average Buyout Price'])
plt.xlabel('Negative Attribute')
plt.ylabel('Average Buyout Price')
plt.title('Impact of Negative Attributes on Buyout Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('negative_attributes_impact.png')
plt.show()

# Create a new DataFrame with only the relevant columns
attribute_price_df = vectisdf[['buyout_price', 'attribute_1', 'attribute_2', 'attribute_3', 'negative_attribute']]

# Function to determine the item type based on the number of positive and negative attributes
def get_item_type(row):
    positive_count = sum(1 for attr in [row['attribute_1'], row['attribute_2'], row['attribute_3']] if attr != '')
    negative_count = 1 if row['negative_attribute'] != '' else 0
    
    if positive_count == 2 and negative_count == 0:
        return '2 Positive, No Negative'
    elif positive_count == 3 and negative_count == 0:
        return '3 Positive, No Negative'
    elif positive_count == 2 and negative_count == 1:
        return '2 Positive, 1 Negative'
    elif positive_count == 3 and negative_count == 1:
        return '3 Positive, 1 Negative'
    else:
        return 'Other'

# Apply the get_item_type function to each row of the DataFrame
attribute_price_df['item_type'] = attribute_price_df.apply(get_item_type, axis=1)

# Calculate the count of each item type
item_type_counts = attribute_price_df['item_type'].value_counts()

# Create a bar graph for item type distribution
plt.figure(figsize=(8, 6))
plt.bar(item_type_counts.index, item_type_counts.values)
plt.xlabel('Item Type')
plt.ylabel('Count')
plt.title('Distribution of Item Types')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('item_type_distribution.png')
plt.show()

# Calculate the average buyout price for each item type
item_type_price = attribute_price_df.groupby('item_type')['buyout_price'].mean()
item_type_price = item_type_price.sort_values(ascending=False)

# Create a bar graph for item type and average buyout price
plt.figure(figsize=(8, 6))
plt.bar(item_type_price.index, item_type_price.values)
plt.xlabel('Item Type')
plt.ylabel('Average Buyout Price')
plt.title('Item Type vs Average Buyout Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('item_type_vs_average_buyout_price.png')
plt.show()

# Print the data as a table
print("Item Type vs Average Buyout Price")
print("----------------------------------")
print("{:<20} {:<20}".format("Item Type", "Average Buyout Price"))
print("----------------------------------")
for item_type, price in item_type_price.items():
    print("{:<20} {:<20.2f}".format(item_type, price))

# Create a scatter plot of buyout_price vs re_rolls
plt.figure(figsize=(10, 6))
plt.scatter(vectisdf['re_rolls'], vectisdf['buyout_price'])

# Calculate the linear regression parameters
slope, intercept, r_value, p_value, std_err = stats.linregress(vectisdf['re_rolls'], vectisdf['buyout_price'])

# Generate points for the regression line
x_vals = np.array(range(vectisdf['re_rolls'].min(), vectisdf['re_rolls'].max() + 1))
y_vals = intercept + slope * x_vals

# Plot the regression line
plt.plot(x_vals, y_vals, color='red', label=f'Regression Line (R={r_value:.2f})')

plt.xlabel('Re-rolls')
plt.ylabel('Buyout Price')
plt.title('Buyout Price vs Re-rolls')
plt.legend()
plt.tight_layout()
plt.savefig('buyout_price_vs_re_rolls.png')
plt.show()

# Calculate the average buyout price for each polarity value
polarity_price = vectisdf.groupby('polarity')['buyout_price'].mean()

# Create a bar plot of average buyout price vs polarity
plt.figure(figsize=(10, 6))
plt.bar(polarity_price.index, polarity_price.values)
plt.xlabel('Polarity')
plt.ylabel('Average Buyout Price')
plt.title('Average Buyout Price vs Polarity')
plt.savefig('average_buyout_price_vs_polarity.png')
plt.tight_layout()
plt.show()


# Function to check if a column contains the "critical_damage" word
def is_critical_damage(column_name):
    return "critical_damage" in column_name.lower()

# Function to get the critical damage value from the corresponding value column
def get_critical_damage(row):
    for i in range(1, 4):
        if is_critical_damage(row[f"attribute_{i}"]):
            return row[f"{i}:value"]
    if is_critical_damage(row["negative_attribute"]):
        return -row["value"]
    return None

# Apply the get_critical_damage function to each row of the DataFrame
vectisdf.loc[:, "critical_damage"] = vectisdf.apply(get_critical_damage, axis=1)

# Extract the non-null critical_damage values from the DataFrame
critical_damage_values = vectisdf[vectisdf['critical_damage'].notnull()]['critical_damage']

# Convert the critical_damage values to numeric
critical_damage_values = pd.to_numeric(critical_damage_values, errors='coerce')

# Remove any NaN or invalid values
critical_damage_values = critical_damage_values[~np.isnan(critical_damage_values)]

# Check if there are any valid critical damage values
if len(critical_damage_values) > 0:
    # Define the bin edges for grouping the values into 5% increments
    bin_edges = np.arange(critical_damage_values.min(), critical_damage_values.max() + 5, 5)

    # Create the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(critical_damage_values, bins=bin_edges, edgecolor='black', align='mid', rwidth=0.8)

    # Set the x-axis tick labels to display the bin ranges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.xticks(bin_centers, [f'{bin_edges[i]:.1f}% - {bin_edges[i+1]:.1f}%' for i in range(len(bin_edges)-1)], rotation=45, ha='right')

    plt.xlabel('Critical Damage Range')
    plt.ylabel('Count')
    plt.title('Distribution of Critical Damage Values')
    plt.tight_layout()
    plt.savefig('critical_damage_distribution.png')
    plt.show()
else:
    print("No valid critical damage values found in the DataFrame.")

