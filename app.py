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


def predict(pdb_code, pdb_file):
    #path_to_pdb = get_pdb(pdb_code=pdb_code, filepath=pdb_file)

    #pdb = open(path_to_pdb, "r").read()
    # switch to misato env if not running from container
    mdh5_file = "inference_for_md.hdf5"
    md_H5File = h5py.File(mdh5_file)

    column_names = ["x", "y", "z", "element"]
    atoms_protein = pd.DataFrame(columns = column_names)
    cutoff = md_H5File["11GS"]["molecules_begin_atom_index"][:][-1] # cutoff defines protein atoms

    atoms_protein["x"] = md_H5File["11GS"]["atoms_coordinates_ref"][:][:cutoff, 0]
    atoms_protein["y"] = md_H5File["11GS"]["atoms_coordinates_ref"][:][:cutoff, 1]
    atoms_protein["z"] = md_H5File["11GS"]["atoms_coordinates_ref"][:][:cutoff, 2]

    atoms_protein["element"] = md_H5File["11GS"]["atoms_element"][:][:cutoff]  

    item = {}
    item["scores"] = 0
    item["id"] = "11GS"
    item["atoms_protein"] = atoms_protein

    transform = GNNTransformMD()
    data_item = transform(item)
    adaptability = model(data_item)
    adaptability = adaptability.detach().numpy()
    
    data = []


    for i in range(adaptability.shape[0]):
        data.append([i, atom_mapping(atoms_protein.iloc[i, atoms_protein.columns.get_loc("element")] - 1), atoms_protein.iloc[i, atoms_protein.columns.get_loc("x")],atoms_protein.iloc[i, atoms_protein.columns.get_loc("y")],atoms_protein.iloc[i, atoms_protein.columns.get_loc("z")],adaptability[i]])

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


demo.launch(debug=True)