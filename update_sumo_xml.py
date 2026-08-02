import xml.etree.ElementTree as ET
import random

def get_random_color():
    return f"{random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)}"

def update_routes():
    tree = ET.parse('/home/nivashini/sumo_project/routes.rou.xml')
    root = tree.getroot()

    # Remove existing vTypes
    for vt in root.findall('vType'):
        root.remove(vt)

    # Add new vTypes
    vtypes = [
        '<vType id="car" vClass="passenger" guiShape="passenger" length="5" maxSpeed="30"/>',
        '<vType id="bus" vClass="bus" guiShape="bus" length="12" maxSpeed="20"/>',
        '<vType id="truck" vClass="truck" guiShape="truck" length="15" maxSpeed="18"/>',
        '<vType id="bike" vClass="motorcycle" guiShape="motorcycle" length="2.5" maxSpeed="25"/>'
    ]
    for vt in reversed(vtypes):
        root.insert(0, ET.fromstring(vt))

    types_list = ["car", "bus", "truck", "bike"]
    weights = [0.7, 0.1, 0.1, 0.1]

    # Assign random types & colors to vehicles
    vehicles = root.findall('vehicle')
    valid_routes = []
    
    for vehicle in vehicles:
        if vehicle.get('type') == 'ambulance':
            continue
        vehicle.set('type', random.choices(types_list, weights)[0])
        vehicle.set('color', get_random_color())
        
        # Collect valid routes for emergency
        route_node = vehicle.find('route')
        if route_node is not None:
            edges = route_node.get('edges')
            if edges and len(edges.split()) > 10:
                valid_routes.append(edges)

    tree.write('/home/nivashini/sumo_project/routes.rou.xml')
    return valid_routes

def update_emergency(valid_routes):
    if not valid_routes:
        print("No long routes found for ambulance. Using defaults.")
        return
        
    chosen_routes = random.sample(valid_routes, min(3, len(valid_routes)))
    
    tree = ET.parse('/home/nivashini/sumo_project/emergency.rou.xml')
    root = tree.getroot()
    
    # Remove existing routes
    for r in root.findall('route'):
        root.remove(r)
        
    # Add new random routes
    for i, edges in enumerate(chosen_routes):
        r = ET.fromstring(f'<route id="ambulance_route_{i}" edges="{edges}"/>')
        root.insert(1, r) # insert right after vType
        
    # Assign the routes to ambulances
    amb_idx = 0
    for v in root.findall('vehicle'):
        if v.get('id').startswith('ambulance'):
            v.set('route', f'ambulance_route_{amb_idx % len(chosen_routes)}')
            amb_idx += 1
            
    tree.write('/home/nivashini/sumo_project/emergency.rou.xml')

def update_pedestrians():
    tree = ET.parse('/home/nivashini/sumo_project/pedestrians.rou.xml')
    root = tree.getroot()
    
    # Check if ped_type exists
    if not root.findall("./vType[@id='ped_type']"):
        # Make pedestrian very large and visible in yellow
        vt = ET.fromstring('<vType id="ped_type" vClass="pedestrian" guiShape="pedestrian" width="2.5" length="2.5" color="255,255,0"/>')
        root.insert(0, vt)
    
    for p in root.findall('person'):
        p.set('type', 'ped_type')
        
    tree.write('/home/nivashini/sumo_project/pedestrians.rou.xml')

if __name__ == "__main__":
    random.seed(42) # For reproducibility
    print("Updating routes...")
    valid_routes = update_routes()
    print(f"Updating emergency tracking with {len(valid_routes)} available routes...")
    update_emergency(valid_routes)
    print("Updating pedestrians to be large and visible...")
    update_pedestrians()
    print("Done XML configurations!")
