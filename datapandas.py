import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the JSON file
with open("vectis_rivens_data.json", "r") as file:
    data = json.load(file)

# Access the riven auctions from the data
auctions = data["payload"]["auctions"]

# Create a list to store the positive attributes
positive_attributes = []

# Process each auction and extract positive attributes
for auction in auctions:
    attributes = auction["item"]["attributes"]
    for attribute in attributes:
        if attribute["positive"]:
            positive_attributes.append(attribute["url_name"])

# Create a DataFrame from the positive attributes
df = pd.DataFrame({"attribute": positive_attributes})

# Define a dictionary to map the current attribute names to the desired names
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

# Replace the attribute names using the mapping dictionary
df["attribute"] = df["attribute"].replace(attribute_mapping)

# Count the frequency of each positive attribute
attribute_counts = df["attribute"].value_counts()
#print(set(df["attribute"]))
# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(attribute_counts.index, attribute_counts.values)
plt.xlabel("Attribute")
plt.ylabel("Frequency")
plt.title("Frequency of Positive Attributes in Vectis Rivens")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot in the same folder
plt.savefig("attribute_frequency_plot.png")

print("Plot saved as attribute_frequency_plot.png")


print(df.describe())
print(df.info())