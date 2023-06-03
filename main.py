
import gradio as gr
import py3Dmol
from Bio.PDB import *

import numpy as np
from Bio.PDB import PDBParser
import pandas as pd
import torch
import os
from MDmodel import GNN_MD
import h5py
from transformMD import GNNTransformMD
import sys
import pytraj as pt
import pickle 

# JavaScript functions
resid_hover = """function(atom,viewer) {{
    if(!atom.label) {{
        atom.label = viewer.addLabel('{0}:'+atom.atom+atom.serial,
            {{position: atom, backgroundColor: 'mintcream', fontColor:'black'}});
    }}
}}"""
hover_func = """
function(atom,viewer) {
    if(!atom.label) {
        atom.label = viewer.addLabel(atom.interaction,
            {position: atom, backgroundColor: 'black', fontColor:'white'});
    }
}"""
unhover_func = """
function(atom,viewer) {
    if(atom.label) {
        viewer.removeLabel(atom.label);
        delete atom.label;
    }
}"""
atom_mapping = {0:'H', 1:'C', 2:'N', 3:'O', 4:'F', 5:'P', 6:'S', 7:'CL', 8:'BR', 9:'I', 10: 'UNK'}

model = GNN_MD(11, 64)
state_dict = torch.load(
    "best_weights_rep0.pt",
    map_location=torch.device("cpu"),
)["model_state_dict"]
model.load_state_dict(state_dict)
model = model.to('cpu')
model.eval()


def run_leap(fileName, path):
    leapText = """
    source leaprc.protein.ff14SB
    source leaprc.water.tip3p
    exp = loadpdb PATH4amb.pdb
    saveamberparm exp PATHexp.top PATHexp.crd
    quit
    """
    with open(path+"leap.in", "w") as outLeap:
        outLeap.write(leapText.replace('PATH', path))	
    os.system("tleap -f "+path+"leap.in >> "+path+"leap.out")

def convert_to_amber_format(pdbName):
    fileName, path = pdbName+'.pdb', ''
    os.system("pdb4amber -i "+fileName+" -p -y -o "+path+"4amb.pdb -l "+path+"pdb4amber_protein.log")
    run_leap(fileName, path)
    traj = pt.iterload(path+'exp.crd', top = path+'exp.top')
    pt.write_traj(path+fileName, traj, overwrite= True)
    print(path+fileName+' was created. Please always use this file for inspection because the coordinates might get translated during amber file generation and thus might vary from the input pdb file.')
    return pt.iterload(path+'exp.crd', top = path+'exp.top')

def get_maps(mapPath):
    residueMap = pickle.load(open(os.path.join(mapPath,'atoms_residue_map_generate.pickle'),'rb'))
    nameMap = pickle.load(open(os.path.join(mapPath,'atoms_name_map_generate.pickle'),'rb'))
    typeMap = pickle.load(open(os.path.join(mapPath,'atoms_type_map_generate.pickle'),'rb'))
    elementMap = pickle.load(open(os.path.join(mapPath,'map_atomType_element_numbers.pickle'),'rb'))
    return residueMap, nameMap, typeMap, elementMap

def get_residues_atomwise(residues):
    atomwise = []
    for name, nAtoms in residues:
        for i in range(nAtoms):
            atomwise.append(name)
    return atomwise

def get_begin_atom_index(traj):
    natoms = [m.n_atoms for m in traj.top.mols]
    molecule_begin_atom_index = [0] 
    x = 0
    for i in range(len(natoms)):
        x += natoms[i]
        molecule_begin_atom_index.append(x)
    print('molecule begin atom index', molecule_begin_atom_index, natoms)
    return molecule_begin_atom_index

def get_traj_info(traj, mapPath):
    coordinates  = traj.xyz
    residueMap, nameMap, typeMap, elementMap = get_maps(mapPath)
    types = [typeMap[a.type] for a in traj.top.atoms]
    elements = [elementMap[typ] for typ in types]
    atomic_numbers = [a.atomic_number for a in traj.top.atoms]
    molecule_begin_atom_index = get_begin_atom_index(traj)
    residues = [(residueMap[res.name], res.n_atoms) for res in traj.top.residues]
    residues_atomwise = get_residues_atomwise(residues)
    return coordinates[0], elements, types, atomic_numbers, residues_atomwise, molecule_begin_atom_index

def write_h5_info(outName, struct, atoms_type, atoms_number, atoms_residue, atoms_element, molecules_begin_atom_index, atoms_coordinates_ref):
    if os.path.isfile(outName):
        os.remove(outName)
    with h5py.File(outName, 'w') as oF:
        subgroup = oF.create_group(struct)     
        subgroup.create_dataset('atoms_residue', data= atoms_residue, compression = "gzip", dtype='i8')
        subgroup.create_dataset('molecules_begin_atom_index', data= molecules_begin_atom_index, compression = "gzip", dtype='i8')
        subgroup.create_dataset('atoms_type', data= atoms_type, compression = "gzip", dtype='i8')
        subgroup.create_dataset('atoms_number', data= atoms_number, compression = "gzip", dtype='i8')  
        subgroup.create_dataset('atoms_element', data= atoms_element, compression = "gzip", dtype='i8')
        subgroup.create_dataset('atoms_coordinates_ref', data= atoms_coordinates_ref, compression = "gzip", dtype='f8')

def preprocess(pdbid: str = None, ouputfile: str = "inference_for_md.hdf5", mask: str = "!@H=", mappath: str = "/maps/"):
    traj = convert_to_amber_format(pdbid)
    atoms_coordinates_ref, atoms_element, atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index = get_traj_info(traj[mask], mappath)
    write_h5_info(ouputfile, pdbid, atoms_type, atoms_number, atoms_residue, atoms_element, molecules_begin_atom_index, atoms_coordinates_ref)

def get_pdb(pdb_code="", filepath=""):
    try:
        return filepath.name
    except AttributeError as e:
        if pdb_code is None or pdb_code == "":
            return None
        else:
            os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
            return f"{pdb_code}.pdb"


def get_offset(pdb):
    pdb_multiline = pdb.split("\n")
    for line in pdb_multiline:
        if line.startswith("ATOM"):
            return int(line[22:27])


def get_pdbid_from_filename(filename: str):
    # Assuming the filename would be of the standard form 11GS.pdb
    return filename.split(".")[0]

def predict(pdb_code, pdb_file):
    #path_to_pdb = get_pdb(pdb_code=pdb_code, filepath=pdb_file)

    #pdb = open(path_to_pdb, "r").read()
    # switch to misato env if not running from container

    pdbid = get_pdbid_from_filename(pdb_file.name)
    mdh5_file = "inference_for_md.hdf5"
    mappath = "/maps"
    mask = "!@H="
    preprocess(pdbid=pdbid, ouputfile=mdh5_file, mask=mask, mappath=mappath)

    md_H5File = h5py.File(mdh5_file)

    column_names = ["x", "y", "z", "element"]
    atoms_protein = pd.DataFrame(columns = column_names)
    cutoff = md_H5File[pdbid]["molecules_begin_atom_index"][:][-1] # cutoff defines protein atoms

    atoms_protein["x"] = md_H5File[pdbid]["atoms_coordinates_ref"][:][:cutoff, 0]
    atoms_protein["y"] = md_H5File[pdbid]["atoms_coordinates_ref"][:][:cutoff, 1]
    atoms_protein["z"] = md_H5File[pdbid]["atoms_coordinates_ref"][:][:cutoff, 2]

    atoms_protein["element"] = md_H5File[pdbid]["atoms_element"][:][:cutoff]  

    item = {}
    item["scores"] = 0
    item["id"] = pdbid
    item["atoms_protein"] = atoms_protein

    transform = GNNTransformMD()
    data_item = transform(item)
    adaptability = model(data_item)
    adaptability = adaptability.detach().numpy()
    
    data = []


    for i in range(10):
        data.append([i, atom_mapping[atoms_protein.iloc[i, atoms_protein.columns.get_loc("element")] - 1], atoms_protein.iloc[topN_ind[i], atoms_protein.columns.get_loc("x")],
        atoms_protein.iloc[topN_ind[i], atoms_protein.columns.get_loc("y")],
        atoms_protein.iloc[topN_ind[i], atoms_protein.columns.get_loc("z")],
        adaptability[topN_ind[i]]
    ])
   
    pdb = open(pdb_file.name, "r").read()

    view = py3Dmol.view(width=600, height=400)
    view.setBackgroundColor('white')
    view.addModel(pdb, "pdb")
    view.setStyle({'stick': {'colorscheme': {'prop': 'resi', 'C': 'turquoise'}}})
   
    for i in range(10):
        view.addSphere({'center':{'x':atoms_protein.iloc[topN_ind[i], atoms_protein.columns.get_loc("x")], 'y':atoms_protein.iloc[topN_ind[i], atoms_protein.columns.get_loc("y")],'z':atoms_protein.iloc[topN_ind[i], atoms_protein.columns.get_loc("z")]},'radius':adaptability[topN_ind[i]]/1.5,'color':'orange','alpha':0.75})    

    view.zoomTo()

    output = view._make_html().replace("'", '"')

    x = f"""<!DOCTYPE html><html> {output} </html>"""  # do not use ' in this input
    return f"""<iframe  style="width: 100%; height:420px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{x}'></iframe>""", pd.DataFrame(data, columns=['index','element','x','y','z','Adaptability'])


callback = gr.CSVLogger()

"""
def predict(pdb_code, pdb_file):
    #path_to_pdb = get_pdb(pdb_code=pdb_code, filepath=pdb_file)

    #pdb = open(path_to_pdb, "r").read()
    # switch to misato env if not running from container

    pdbid = get_pdbid_from_filename(pdb_file.name)
    mdh5_file = "inference_for_md.hdf5"
    mappath = "/maps"
    mask = "!@H="
    preprocess(pdbid=pdbid, ouputfile=mdh5_file, mask=mask, mappath=mappath)

    md_H5File = h5py.File(mdh5_file)

    column_names = ["x", "y", "z", "element"]
    atoms_protein = pd.DataFrame(columns = column_names)
    cutoff = md_H5File[pdbid]["molecules_begin_atom_index"][:][-1] # cutoff defines protein atoms

    atoms_protein["x"] = md_H5File[pdbid]["atoms_coordinates_ref"][:][:cutoff, 0]
    atoms_protein["y"] = md_H5File[pdbid]["atoms_coordinates_ref"][:][:cutoff, 1]
    atoms_protein["z"] = md_H5File[pdbid]["atoms_coordinates_ref"][:][:cutoff, 2]

    atoms_protein["element"] = md_H5File[pdbid]["atoms_element"][:][:cutoff]  

    item = {}
    item["scores"] = 0
    item["id"] = pdbid
    item["atoms_protein"] = atoms_protein

    transform = GNNTransformMD()
    data_item = transform(item)
    adaptability = model(data_item)
    adaptability = adaptability.detach().numpy()
    
    data = []


    for i in range(adaptability.shape[0]):
        data.append([i, atom_mapping[atoms_protein.iloc[i, atoms_protein.columns.get_loc("element")] - 1], atoms_protein.iloc[i, atoms_protein.columns.get_loc("x")],atoms_protein.iloc[i, atoms_protein.columns.get_loc("y")],atoms_protein.iloc[i, atoms_protein.columns.get_loc("z")],adaptability[i]])

    topN = 100
    topN_ind = np.argsort(adaptability)[::-1][:topN]    

    pdb = open(pdb_file.name, "r").read()

    view = py3Dmol.view(width=600, height=400)
    view.setBackgroundColor('white')
    view.addModel(pdb, "pdb")
    view.setStyle({'stick': {'colorscheme': {'prop': 'resi', 'C': 'turquoise'}}})
   
    for i in range(topN):
        view.addSphere({'center':{'x':atoms_protein.iloc[topN_ind[i], atoms_protein.columns.get_loc("x")], 'y':atoms_protein.iloc[topN_ind[i], atoms_protein.columns.get_loc("y")],'z':atoms_protein.iloc[topN_ind[i], atoms_protein.columns.get_loc("z")]},'radius':adaptability[topN_ind[i]]/1.5,'color':'orange','alpha':0.75})    

    view.zoomTo()

    output = view._make_html().replace("'", '"')

    x = f"""<!DOCTYPE html><html> {output} </html>"""  # do not use ' in this input
    return f"""<iframe  style="width: 100%; height:420px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{x}'></iframe>""", pd.DataFrame(data, columns=['index','element','x','y','z','Adaptability'])


callback = gr.CSVLogger()
"""

def run():
    with gr.Blocks() as demo:
        gr.Markdown("# Protein Adaptability Prediction")
        
        #text_input = gr.Textbox()
        #text_output = gr.Textbox()
        #text_button = gr.Button("Flip")
        inp = gr.Textbox(placeholder="PDB Code or upload file below", label="Input structure")
        pdb_file = gr.File(label="PDB File Upload")
        #with gr.Row():
        #    helix = gr.ColorPicker(label="helix")
        #    sheet = gr.ColorPicker(label="sheet")
        #    loop = gr.ColorPicker(label="loop")
        single_btn = gr.Button(label="Run")
        with gr.Row():
            html = gr.HTML()
        with gr.Row():
            dataframe = gr.Dataframe()
                
        single_btn.click(fn=predict, inputs=[inp, pdb_file], outputs=[html, dataframe])


    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    run()
