import requests
import json
from ratelimit import limits, sleep_and_retry

# Set the base URL for the Warframe Market API
base_url = "https://api.warframe.market/v1"

# Set the endpoint for searching riven auctions
endpoint = "/auctions/search"

# Set the headers for the request
headers = {
    "Platform": "pc",
    "Accept": "application/json"
}
weaponslist = ['kulstar', 'heliocor', 'nagantaka', 'ocucor', 'falcor', 'paracesis', 'exergis', 'battacor', 'euphona_prime', 'fusilai', 'lato', 'magnus', 'ack_and_brunt', 'amphis', 'anku', 'arca_titron', 'balla', 'boltace', 'broken_war', 'caustacyst', 'cerata', 'hate', 'soma', 'synapse', 'tigris', 'tonkor', 'torid', 'vectis', 'veldt', 'vulkar', 'zarr', 'zenith', 'zhuge', 'acrid', 'afuris', 'akbolto', 'akbronco', 'akjagara', 'aklato', 'arca_scisco', 'cycron', 'detron', 'embolist', 'dual_kamas', 'kuva_shildeg', 'kuva_ayanga', 'vermisplicer', 'sporelacer', 'argonak', 'castanas', 'cestra', 'dual_toxocyst', 'hystrix', 'fragor', 'galatine', 'gazal_machete', 'gram', 'guandao', 'gunsen', 'dual_skana', 'jaw_sword', 'kesheg', 'kestrel', 'kronsh', 'ballistica', 'heat_dagger', 'jat_kittag', 'sonicor', 'tombfinger', 'gaze', 'pupacyst', 'arca_plasmor', 'galvacord', 'bronco', 'dual_cestra', 'furis', 'ankyros', 'atterax', 'bo', 'cassowar', 'ceramic_dagger', 'halikar', 'hirudo', 'mire', 'twin_krohkur', 'latron', 'lenz', 'miter', 'mutalist_cernos', 'mutalist_quanta', 'opticor', 'panthera', 'penta', 'phage', 'phantasma', 'quanta', 'rubico', 'simulor', 'snipetron', 'sobek', 'stradavar', 'strun', 'sybaris', 'tetra', 'gammacor', 'nukor', 'pandero', 'pox', 'pyrana', 'sicarus', 'spira', 'dehtat', 'destreza', 'dex_dakra', 'dokrahm', 'dragon_nikana', 'dual_cleavers', 'dual_ether', 'dual_heat_swords', 'dual_ichor', 'dual_keres', 'kama', 'kogake', 'multron', 'vulcax', 'masseter', 'pathocyst', 'kuva_chakkhurr', 'lecta', 'machete', 'magistar', 'mewan', 'mios', 'rabvee', 'war', 'artax', 'burst_laser', 'deth_machine_rifle', 'laser_rifle', 'astilla', 'baza', 'boar', 'flux_rifle', 'cronus', 'endura', 'glaive', 'heat_sword', 'jat_kusar', 'nami_skyla', 'nami_solo', 'nikana', 'ninkondi', 'obex', 'ohma', 'okina', 'ooltha', 'orthos', 'orvius', 'pangolin_sword', 'plague_keewar', 'plague_kripath', 'plasma_sword', 'prova', 'redeemer', 'skana', 'harpak', 'hema', 'hind', 'ignis', 'javlok', 'karak', 'kohm', 'lanka', 'paris', 'quartakk', 'scourge', 'spectra', 'staticor', 'stubba', 'stug', 'talons', 'twin_grakatas', 'twin_gremlins', 'twin_kohmak', 'twin_rogga', 'twin_vipers', 'tysis', 'vasto', 'viper', 
'zylok', 'dakra_prime', 'dark_dagger', 'kuva_twin_stubbas', 'pennant', 'kuva_bramma', 'stropha', 'morgha', 'catabolyst', 'hikou', 'knell', 'kohmak', 'kraken', 'kunai', 'lex', 'marelok', 'broken_scepter', 'dual_zoren', 
'ether_daggers', 'ether_reaper', 'ether_sword', 'fang', 'furax', 'karyst', 'stinger', 'vulklok', 'catchmoon', 'plinx', 'korrudo', 'fulmin', 'cobra_and_crane', 'komorex', 'cyanex', 'tatsu', 'cyngas', 'larkspur', 'imperator', 'corvas', 'phaedra', 'grattler', 'dual_decurion', 'attica', 'boltor', 'braton', 'corinth', 'ferrox', 'hek', 'ogris', 'paracyst', 'supra', 'tenora', 'tiberon', 'aklex', 'akmagnus', 'aksomati', 'akstiletto', 'akvasto', 'akzani', 'angstrum', 'atomos', 'azima', 'bolto', 'brakk', 'despair', 'dual_raza', 'reaper_prime', 'ripkas', 'dark_sword', 'kreska', 'velocitus', 'fluctus', 'quatz', 'acceltra', 'akarius', 'tazicor', 'cryotra', 'shedu', 'cortege', 'vitrica', 'pulmonars', 'bubonico', 'proboscis_cernos', 'sporothrix', 'arum_spinosa', 'sarpa', 'scindo', 'scoliac', 'sepfahn', 'serro', 'shaku', 'sheev', 'sibear', 'sigma_and_octantis', 'silva_and_aegis', 'skiajati', 'sydon', 'tekko', 'tipedo', 'tonbo', 'twin_basolk', 'amprex', 'burston', 'buzlok', 'cernos', 'convectrix', 'daikyu', 'dera', 'drakgoon', 'dread', 'glaxion', 'gorgon', 'grakata', 'grinlok', 'seer', 'zakti', 'cyath', 'dark_split_sword_(dual_swords)', 'krohkur', 'kronen', 'lacera', 'lesion', 'venka', 'volnus', 'zenistar', 'deconstructor', 'sweeper', 'rattleguts', 'wolf_sledge', 'quellor', 'basmu', 'stahlta', 'velox', 'helstrum', 'xoris', 'athodai', 'sepulcrum', 'quassus', 'trumna', 'zymos', 'keratinos', 'mausolon', 'cedo', 'verglas', 'epitaph', 'tenet_envoy', 'tenet_spirex', 'kompressa', 'tenet_exec', 'tenet_grigori', 'ambassador', 
'cadus', 'tenet_livia', 'tenet_diplos', 'tenet_agendus', 'lacerten', 'akaten', 'batoten', 'vastilok', 'ghoulsaw', 'verdilac', 'nepheri', 'rumblejack', 'korumm', 'venato', 'nataruk', 'hespar', 'alternox', 'aeolak', 'phenmor', 'laetum', 'praedos', 'innodem', 'felarx', 'vericres', 'slaytra', 'aegrit', 'afentis', 'sarofang', 'perigale', 'corufell', 'steflos', 'sampotes', 'azothane', 'syam', 'cinta', 'sun_and_moon', 'edun', 'rauta', 'argo_and_vel', 'gotva_prime', 'dorrclave', 'ekhein', 'mandonel', 'grimoire', 'onos', 'ruvox']
# Define the rate limit (3 requests per second)
@sleep_and_retry
@limits(calls=3, period=1)
def make_request(url, params, headers):
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code: {response.status_code}")
    return response.json()

counter = 9
stats = 'electric_damage'
while True:
    # Load existing data from the JSON file if it exists
    try:
        with open("all_rivens_data.json", "r") as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {"payload": {"auctions": []}}


    # Abomination of a tempory if statement. I foundout at the end that every request is limited to pull 500 possible data points (my graphs looked weird)   
    # I think a better solution would be to use a dictionary to map the counter onto the stats values
        
    # if counter == 1:
    #     stats = 'damage_vs_grineer'
    # elif counter == 2:
    #     stats = 'damage_vs_corpus'
    # elif counter == 3:
    #     stats = 'damage_vs_infested'
    # elif counter == 4:
    #     stats = 'range'
    # elif counter == 5:
    #     stats = 'combo_duration'
    # elif counter == 6:
    #     stats = 'critical_chance'
    # elif counter == 7:
    #     stats = 'critical_chance_on_slide_attack'
    # elif counter == 8:
    #     stats = 'critical_damage'
    # elif counter == 9:
    #     stats = 'base_damage_/_melee_damage'
    if counter == 10:
        stats = 'reload_speed'
    elif counter == 11:
        stats = 'recoil'
    elif counter == 12:
        stats = 'finisher_damage'
    elif counter == 13:
        stats = 'punch_through'
    elif counter == 14:
        stats = 'projectile_speed'
    elif counter == 15:
        stats = 'puncture_damage'
    elif counter == 16:
        stats = 'multishot'
    elif counter == 17:
        stats = 'punch_through'
    elif counter == 18:
        stats = 'puncture_damage'
    elif counter == 19:
        stats = 'reload_speed'
    elif counter == 20:
        stats = 'status_chance'
    elif counter == 21:
        stats = 'recoil'
    elif counter == 22:
        stats = 'chance_to_gain_combo_count'
    
        
        
    
    # Create a set to store the IDs of existing auctions
    existing_auction_ids = {auction["id"] for auction in existing_data["payload"]["auctions"]}
    
    # Make a request for each weapon in the weaponslist
    #Stats are only added to avoid limit of total data retreived
    for weapon in weaponslist:
        params = {
            "type": "riven",
            "weapon_url_name": weapon,
            "positive_stats" : stats,
        }
        
        # Make the GET request to the API
        new_data = make_request(base_url + endpoint, params, headers)
        
        # Filter out duplicate auctions based on the "id" field
        unique_new_auctions = [auction for auction in new_data["payload"]["auctions"] if auction["id"] not in existing_auction_ids]
        
        # Add the unique new auctions to the existing data
        existing_data["payload"]["auctions"] += unique_new_auctions
        
        # Update the set of existing auction IDs
        existing_auction_ids.update(auction["id"] for auction in unique_new_auctions)

    # Save the updated data to the JSON file
    with open("all_rivens_data.json", "w") as file:
        json.dump(existing_data, file, indent=4)

    print("Data appended to all_rivens_data.json and duplicates removed")
    #tempory counter to cycle Stats values
    counter += 1