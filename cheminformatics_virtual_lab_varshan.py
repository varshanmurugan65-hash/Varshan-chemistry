import os
import tempfile
from io import BytesIO

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import py3Dmol
import streamlit.components.v1 as components

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)


# ============================================================
# PAGE CONFIGURATION AND STYLING
# ============================================================

st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.section-title {
    font-size: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DEFAULT STUDENT DETAILS
# ============================================================

DEFAULT_STUDENT_NAME = "Varshan"
DEFAULT_REGISTRATION_NUMBER = "RA2611026050285"

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "CCO"
if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Ethanol"
if "student_name" not in st.session_state:
    st.session_state.student_name = DEFAULT_STUDENT_NAME
if "registration_number" not in st.session_state:
    st.session_state.registration_number = DEFAULT_REGISTRATION_NUMBER


def load_shared_example():
    st.session_state.active_smiles = EXAMPLE_MOLECULES[
        st.session_state.selected_example
    ]


# ============================================================
# MOLECULE HELPERS
# ============================================================

def get_molecule(smiles):
    if not smiles or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None


def calculate_bond_information(mol):
    """Calculate heavy-atom bonds, sigma bonds and pi bonds."""
    heavy_atom_bonds = mol.GetNumBonds()
    kekule_mol = Chem.Mol(mol)

    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        pass

    pi_bonds = 0
    for bond in kekule_mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bond_type == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bond_type == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            pi_bonds += 0.5

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()

    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": sigma_bonds,
        "Pi Bonds": pi_bonds,
        "Total Bonds": sigma_bonds + pi_bonds,
        "Total Atoms Including H": mol_with_h.GetNumAtoms(),
    }


def calculate_properties(mol):
    bond_info = calculate_bond_information(mol)
    return {
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": round(Descriptors.MolWt(mol), 3),
        "Exact Molecular Weight": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),
        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Ring Count": Lipinski.RingCount(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Number of Atoms": mol.GetNumAtoms(),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Heavy Atom Bonds": bond_info["Heavy Atom Bonds"],
        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],
        "Pi (π) Bonds": bond_info["Pi Bonds"],
        "Total Bonds (σ + π)": bond_info["Total Bonds"],
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "Molar Refractivity": round(Crippen.MolMR(mol), 3),
    }


def generate_2d_image(mol):
    return Draw.MolToImage(mol, size=(700, 500))


def generate_3d_molecule(mol):
    try:
        mol3d = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol3d, params) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol3d)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol3d)
            except Exception:
                pass
        return mol3d
    except Exception:
        return None


def display_3d_structure(mol3d, style, show_surface=False):
    view = py3Dmol.view(width=900, height=600)
    view.addModel(Chem.MolToMolBlock(mol3d), "mol")

    if style == "Ball and Stick":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.30}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.20}})
    elif style == "Space Filling":
        view.setStyle({"sphere": {"scale": 1.0}})
    elif style == "Wireframe":
        view.setStyle({"line": {"linewidth": 2}})
    else:
        view.setStyle({"stick": {}, "sphere": {"scale": 0.20}})

    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.7})
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=620)


# ============================================================
# PDF REPORT HELPERS
# ============================================================

def generate_molecule_pdf(input_smiles, mol, properties):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=45, rightMargin=45,
        topMargin=45, bottomMargin=45
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=20,
        alignment=TA_CENTER, textColor=HexColor("#17365D")
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading1"], fontSize=14,
        textColor=HexColor("#17365D")
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["BodyText"], fontSize=10,
        leading=14, alignment=TA_JUSTIFY
    )

    story = [
        Paragraph("MOLECULAR PROPERTIES REPORT", title_style),
        Spacer(1, 0.3 * inch),
        Paragraph("1. Molecular Information", heading_style),
    ]
    basic_data = [
        ["Property", "Value"],
        ["Input SMILES", input_smiles],
        ["Canonical SMILES", Chem.MolToSmiles(mol)],
        ["Molecular Formula", properties["Molecular Formula"]],
    ]
    basic_table = Table(basic_data, colWidths=[2 * inch, 4 * inch])
    basic_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [basic_table, Spacer(1, 0.3 * inch),
              Paragraph("2. Molecular Properties", heading_style)]

    property_data = [["Property", "Value"]]
    property_data.extend([[str(k), str(v)] for k, v in properties.items()])
    property_table = Table(property_data, colWidths=[3 * inch, 3 * inch], repeatRows=1)
    property_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [
        property_table, Spacer(1, 0.3 * inch),
        Paragraph("3. Interpretation", heading_style),
        Paragraph(
            "This report was generated by the Cheminformatics Virtual Laboratory. "
            "Molecular properties were calculated from the submitted SMILES notation "
            "using the RDKit cheminformatics toolkit.",
            body_style,
        ),
    ]
    doc.build(story)
    result = buffer.getvalue()
    buffer.close()
    return result


def save_2d_structure_png(mol):
    image = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    image.save(tmp.name)
    return tmp.name


def save_3d_structure_png(mol):
    """Render a 3D ball-and-stick projection for the final PDF."""
    import numpy as np

    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None

    conf = mol3d.GetConformer()
    coords = np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
        for i in range(mol3d.GetNumAtoms())
    ], dtype=float)

    centered = coords - coords.mean(axis=0)
    if len(coords) >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]

    cpk = {
        "H": "#FFFFFF", "C": "#3F3F3F", "N": "#2F5BFF",
        "O": "#E53935", "F": "#65C466", "P": "#F39C12",
        "S": "#F4D03F", "Cl": "#27AE60", "Br": "#8E3B2F",
        "I": "#8E44AD",
    }
    sizes = {"H": 45, "C": 200, "N": 230, "O": 240, "F": 220,
             "P": 250, "S": 260, "Cl": 260, "Br": 280, "I": 300}

    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    ax.set_facecolor("white")
    for bond in mol3d.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        x1, y1 = projected[i]
        x2, y2 = projected[j]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue
        px, py = -dy / length, dx / length
        bond_type = bond.GetBondType()
        offsets = (-0.035, 0.035) if bond_type == Chem.BondType.DOUBLE else (
            (-0.055, 0.0, 0.055) if bond_type == Chem.BondType.TRIPLE else (0.0,)
        )
        for off in offsets:
            ax.plot([x1 + px * off, x2 + px * off],
                    [y1 + py * off, y2 + py * off],
                    color="#707070", linewidth=2.5, solid_capstyle="round", zorder=1)

    for i, atom in enumerate(mol3d.GetAtoms()):
        symbol = atom.GetSymbol()
        x, y = projected[i]
        ax.scatter(x, y, s=sizes.get(symbol, 200), c=cpk.get(symbol, "#A0A0A0"),
                   edgecolors="#222222", linewidths=1.0, zorder=3)
        if symbol not in ("H", "C"):
            ax.text(x, y, symbol, ha="center", va="center", fontsize=8,
                    fontweight="bold", color="black", zorder=4)

    xmin, ymin = projected.min(axis=0)
    xmax, ymax = projected.max(axis=0)
    pad = 0.22 * max(xmax - xmin, ymax - ymin, 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("3D Molecular Structure (Ball-and-Stick)", fontsize=16, pad=16)
    fig.tight_layout(pad=1.0)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=260, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    return tmp.name


def get_property_interpretation(name, value):
    if name == "Molecular Formula":
        return "Shows the elemental composition of the molecule."
    if name in ("Molecular Weight", "Exact Molecular Weight"):
        return "Describes the molecular mass; exact mass uses exact isotopic masses."
    if name == "LogP":
        if value < 0:
            return "Indicates preference for aqueous/polar environments."
        if value < 1:
            return "Indicates low lipophilicity."
        if value < 3:
            return "Indicates moderate lipophilicity."
        return "Indicates relatively high lipophilicity."
    if name == "TPSA":
        return "Estimates the polar surface area contributed by polar atoms and groups."
    if name == "Hydrogen Bond Donors":
        return f"The molecule has {value} hydrogen-bond donor site(s)."
    if name == "Hydrogen Bond Acceptors":
        return f"The molecule has {value} hydrogen-bond acceptor site(s)."
    if name == "Rotatable Bonds":
        return f"{value} rotatable bond(s) indicate molecular flexibility."
    if name == "Ring Count":
        return f"The molecule contains {value} ring(s)."
    if name == "Aromatic Rings":
        return f"The molecule contains {value} aromatic ring(s)."
    if name == "Aliphatic Rings":
        return f"The molecule contains {value} non-aromatic ring(s)."
    if name in ("Number of Atoms", "Heavy Atoms"):
        return "Provides a count of atoms in the molecular representation."
    if name == "Heavy Atom Bonds":
        return "Counts bonds between non-hydrogen atoms."
    if name == "Sigma (σ) Bonds":
        return "Counts sigma bonds, including bonds to hydrogens added by RDKit."
    if name == "Pi (π) Bonds":
        return "Estimates pi bonds from multiple bonds and aromatic representation."
    if name == "Total Bonds (σ + π)":
        return "Sum of the calculated sigma and pi bond counts."
    if name == "Fraction Csp3":
        return "Indicates the fraction of carbon atoms with sp3 character."
    if name == "Molar Refractivity":
        return "Relates to molecular volume and electronic polarizability."
    return "A computed descriptor of the molecular structure."


def build_key_interpretation_points(properties):
    return [
        f"<b>Molecular identity:</b> Formula {properties['Molecular Formula']}; "
        f"molecular weight {properties['Molecular Weight']} g/mol.",
        f"<b>Lipophilicity:</b> LogP = {properties['LogP']}, describing preference "
        "between polar and non-polar environments.",
        f"<b>Polarity:</b> TPSA = {properties['TPSA']} Å².",
        f"<b>Hydrogen bonding:</b> {properties['Hydrogen Bond Donors']} donor(s) and "
        f"{properties['Hydrogen Bond Acceptors']} acceptor(s).",
        f"<b>Flexibility:</b> {properties['Rotatable Bonds']} rotatable bond(s).",
        f"<b>Ring character:</b> {properties['Ring Count']} ring(s), including "
        f"{properties['Aromatic Rings']} aromatic ring(s).",
        f"<b>Bonding:</b> {properties['Sigma (σ) Bonds']} sigma bonds and "
        f"{properties['Pi (π) Bonds']} pi bonds (calculated estimate).",
        f"<b>Three-dimensional character:</b> Fraction Csp3 = {properties['Fraction Csp3']}.",
        "<b>Conclusion:</b> Interpret these descriptors together with the 2D connectivity "
        "and generated 3D geometry. Computational values are useful for learning and comparison.",
    ]


def generate_final_report(student_name, registration_number, input_smiles, mol, properties):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=38, rightMargin=38,
        topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ProjectTitle", parent=styles["Title"], fontSize=20,
        leading=25, alignment=TA_CENTER, textColor=HexColor("#17365D")
    )
    heading_style = ParagraphStyle(
        "ProjectHeading", parent=styles["Heading1"], fontSize=14,
        leading=18, textColor=HexColor("#17365D"), spaceBefore=8, spaceAfter=8
    )
    body_style = ParagraphStyle(
        "ProjectBody", parent=styles["BodyText"], fontSize=9.5,
        leading=13, alignment=TA_JUSTIFY
    )
    small_style = ParagraphStyle(
        "SmallTableText", parent=styles["BodyText"], fontSize=7.5, leading=9.5
    )
    header_style = ParagraphStyle(
        "HeaderTableText", parent=styles["BodyText"], fontSize=8,
        leading=9, textColor=colors.white, alignment=TA_CENTER
    )

    story = [
        Spacer(1, 0.25 * inch),
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Spacer(1, 0.12 * inch),
        Paragraph("VARSHAN'S FINAL MOLECULAR ANALYSIS REPORT", heading_style),
        Spacer(1, 0.18 * inch),
    ]

    student_data = [
        ["Student Name", student_name or DEFAULT_STUDENT_NAME],
        ["Registration Number", registration_number or DEFAULT_REGISTRATION_NUMBER],
        ["Input SMILES", input_smiles],
        ["Canonical SMILES", Chem.MolToSmiles(mol)],
        ["Molecular Formula", properties["Molecular Formula"]],
    ]
    student_table = Table(student_data, colWidths=[2.0 * inch, 4.7 * inch])
    student_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF0F8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [student_table, Spacer(1, 0.22 * inch),
              Paragraph("1. Molecular Structures", heading_style)]

    img2d = save_2d_structure_png(mol)
    img3d = save_3d_structure_png(mol)
    try:
        if img2d and img3d:
            structure_table = Table(
                [[Image(img2d, width=3.15 * inch, height=2.25 * inch),
                  Image(img3d, width=3.15 * inch, height=2.25 * inch)]],
                colWidths=[3.3 * inch, 3.3 * inch],
            )
            structure_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            story += [
                structure_table, Spacer(1, 0.08 * inch),
                Paragraph(
                    "<b>Figure 1.</b> Left: 2D molecular connectivity. Right: generated "
                    "3D ball-and-stick geometry.",
                    body_style,
                ),
            ]
        elif img2d:
            story.append(Image(img2d, width=5.8 * inch, height=4.1 * inch))
    except Exception:
        pass

    story += [Spacer(1, 0.18 * inch),
              Paragraph("2. Complete Molecular Properties and Interpretation", heading_style)]
    prop_data = [[
        Paragraph("<b>Property</b>", header_style),
        Paragraph("<b>Value</b>", header_style),
        Paragraph("<b>Interpretation</b>", header_style),
    ]]
    for key, value in properties.items():
        prop_data.append([
            Paragraph(str(key), small_style),
            Paragraph(str(value), small_style),
            Paragraph(get_property_interpretation(key, value), small_style),
        ])
    property_table = Table(
        prop_data, colWidths=[1.75 * inch, 1.0 * inch, 3.85 * inch], repeatRows=1
    )
    property_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [property_table, Spacer(1, 0.18 * inch),
              Paragraph("3. Point-wise Molecular Interpretation", heading_style),
              Paragraph(
                  "The following points summarize structural and physicochemical "
                  "features calculated from the submitted SMILES.",
                  body_style,
              ),
              Spacer(1, 0.08 * inch)]
    for point in build_key_interpretation_points(properties):
        story += [Paragraph(point, body_style), Spacer(1, 0.06 * inch)]
    story += [
        Spacer(1, 0.12 * inch),
        Paragraph("4. Learning Note", heading_style),
        Paragraph(
            "The numerical descriptors are calculated from the submitted structure. "
            "They are useful for molecular comparison and cheminformatics learning; "
            "experimental measurements can differ with conditions and methods.",
            body_style,
        ),
    ]

    try:
        doc.build(story)
        return buffer.getvalue()
    finally:
        buffer.close()
        for path in (img2d, img3d):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧪 Virtual Lab")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📖 Theory",
        "🧬 Molecular Visualization",
        "📊 Molecular Properties Report",
        "📈 Structure–Property Analysis",
        "📝 Assessment",
        "📄 Final Project Report",
    ],
)
st.sidebar.divider()
st.sidebar.markdown("### 👨‍🎓 Student Information")
st.sidebar.text_input("Student Name", key="student_name")
st.sidebar.text_input("Registration Number", key="registration_number")
st.sidebar.markdown("""
### 🎓 Learning Objectives

Students can learn:
- SMILES notation
- Molecular representation
- 2D visualization
- 3D visualization
- Molecular descriptors
- Structure-property relationships
""")


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":
    logo_path = os.path.join(os.path.dirname(__file__), "srm_logo.jpeg")
    if os.path.exists(logo_path):
        logo_col1, logo_col2, logo_col3 = st.columns([1, 4, 1])
        with logo_col2:
            st.image(logo_path, use_container_width=True)

    st.markdown(
        '<div class="main-title">🧪 Cheminformatics Virtual Laboratory</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style="text-align:center; font-size:24px; font-weight:700; color:#1f4e79;">
    SRM Institute of Science and Technology, Tiruchirappalli
    </div>
    <div style="text-align:center; font-size:18px; font-weight:600; margin-top:5px;">
    Faculty of Engineering and Technology | School of Sciences | Division of Chemistry
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("## Welcome to the Virtual Laboratory")
    st.write(
        "This interactive virtual laboratory helps students understand important "
        "concepts in cheminformatics and computational chemistry."
    )
    st.markdown("### 🧪 What can students do?")
    st.markdown("""
    - ✅ Enter and validate SMILES notation
    - ✅ Generate 2D molecular structures
    - ✅ Generate interactive 3D structures
    - ✅ Select different molecular visualization styles
    - ✅ Calculate molecular properties
    - ✅ Download molecular reports
    - ✅ Compare multiple molecules
    - ✅ Complete an assessment
    """)
    st.markdown("### Example Molecules")
    example_df = pd.DataFrame({
        "Molecule": list(EXAMPLE_MOLECULES.keys()),
        "SMILES": list(EXAMPLE_MOLECULES.values()),
    })
    st.dataframe(example_df, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("### 👨‍🎓 Student Details")
    st.markdown(f"**Student Name:** {st.session_state.student_name or DEFAULT_STUDENT_NAME}")
    st.markdown(
        f"**Registration Number:** "
        f"{st.session_state.registration_number or DEFAULT_REGISTRATION_NUMBER}"
    )
    st.markdown("**Department:** AIML")
    st.markdown(
        "**Institution:** SRM Institute of Science and Technology, Tiruchirappalli"
    )


# ============================================================
# THEORY PAGE
# ============================================================

elif page == "📖 Theory":
    st.title("📖 Introduction to Cheminformatics")
    st.markdown("""
## What is Cheminformatics?
Cheminformatics combines chemistry, computer science and data analysis. It uses
computational techniques to store, process, visualize and analyze chemical information.

## SMILES
**SMILES** means **Simplified Molecular Input Line Entry System**. It represents a
molecular structure using a text string.

| Molecule | SMILES |
|---|---|
| Ethanol | `CCO` |
| Benzene | `c1ccccc1` |
| Acetic Acid | `CC(=O)O` |

## Molecular Descriptors
Molecular descriptors provide numerical information about a chemical structure.
Examples include molecular weight, LogP, TPSA, hydrogen-bond donors, hydrogen-bond
acceptors and rotatable bonds.

## 3D Molecular Models
- **Ball and Stick:** shows atoms as spheres and bonds as sticks.
- **Stick Model:** emphasizes chemical bonding.
- **Space Filling:** shows approximate molecular volume.
- **Wireframe:** provides a simplified representation.
""")


# ============================================================
# MOLECULAR VISUALIZATION PAGE
# ============================================================

elif page == "🧬 Molecular Visualization":
    st.title("🧬 Molecular Visualization Laboratory")
    st.info("Enter a SMILES or load an example. The molecule is used across laboratory pages.")
    col1, col2 = st.columns([2, 1])
    with col1:
        smiles = st.text_input("Enter SMILES", key="active_smiles")
    with col2:
        st.selectbox("Choose Example Molecule", list(EXAMPLE_MOLECULES.keys()),
                     key="selected_example")
        st.button("Load Example", on_click=load_shared_example,
                  key="load_visualization_example")

    mol = get_molecule(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES notation.")
    else:
        st.success("✅ Valid Molecular Structure!")
        st.session_state.latest_smiles = smiles
        st.session_state.latest_properties = calculate_properties(mol)
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("2D Molecular Structure")
            st.image(generate_2d_image(mol))
        with col2:
            st.subheader("Molecular Information")
            st.write("**Input SMILES:**", smiles)
            st.write("**Canonical SMILES:**", Chem.MolToSmiles(mol))
            st.write("**Formula:**", rdMolDescriptors.CalcMolFormula(mol))
            bond_info = calculate_bond_information(mol)
            st.write("**Heavy Atoms:**", mol.GetNumHeavyAtoms())
            st.write("**Total Atoms (including H):**", bond_info["Total Atoms Including H"])
            st.write("**Heavy-Atom Bonds:**", bond_info["Heavy Atom Bonds"])
            st.write("**Sigma (σ) Bonds:**", bond_info["Sigma Bonds"])
            st.write("**Pi (π) Bonds:**", bond_info["Pi Bonds"])
            st.write("**Total Bonds (σ + π):**", bond_info["Total Bonds"])

        st.divider()
        st.subheader("🌐 Interactive 3D Molecular Structure")
        style = st.selectbox(
            "Select Molecular Model",
            ["Ball and Stick", "Stick", "Space Filling", "Wireframe", "Cartoon"],
            key="visualization_style",
        )
        show_surface = st.checkbox("Show van der Waals Surface", key="visualization_surface")
        if st.button("🚀 Generate 3D Structure", key="visualization_generate_3d"):
            with st.spinner("Generating 3D molecular structure..."):
                mol3d = generate_3d_molecule(mol)
            if mol3d is None:
                st.error("Unable to generate the 3D structure.")
            else:
                display_3d_structure(mol3d, style, show_surface)


# ============================================================
# MOLECULAR PROPERTIES REPORT
# ============================================================

elif page == "📊 Molecular Properties Report":
    st.title("📊 Complete Molecular Properties Report")
    st.info("Enter a valid SMILES to generate structures and molecular properties.")
    col1, col2 = st.columns([2, 1])
    with col1:
        smiles = st.text_input("Enter New SMILES", key="active_smiles")
    with col2:
        st.selectbox("Choose Example Molecule", list(EXAMPLE_MOLECULES.keys()),
                     key="selected_example")
        st.button("Load Example", on_click=load_shared_example,
                  key="load_properties_example")

    st.caption("The same molecule is used for the structures, property table and downloads.")
    if smiles:
        mol = get_molecule(smiles)
        if mol is None:
            st.error("❌ Invalid SMILES notation. Please check the structure.")
        else:
            st.success("✅ Valid Molecular Structure Detected!")
            properties = calculate_properties(mol)
            canonical_smiles = Chem.MolToSmiles(mol)
            st.session_state.latest_smiles = smiles
            st.session_state.latest_properties = properties

            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.subheader("🧬 2D Molecular Structure")
                st.image(generate_2d_image(mol))
            with col2:
                st.subheader("📋 Basic Information")
                st.write("**Input SMILES:**", smiles)
                st.write("**Canonical SMILES:**", canonical_smiles)
                st.write("**Molecular Formula:**", properties["Molecular Formula"])
                st.write("**Atoms:**", properties["Number of Atoms"])
                st.write("**Heavy Atoms:**", properties["Heavy Atoms"])
                st.write("**Heavy-Atom Bonds:**", properties["Heavy Atom Bonds"])
                st.write("**Sigma (σ) Bonds:**", properties["Sigma (σ) Bonds"])
                st.write("**Pi (π) Bonds:**", properties["Pi (π) Bonds"])
                st.write("**Total Bonds (σ + π):**", properties["Total Bonds (σ + π)"])

            st.divider()
            st.subheader("📊 Important Molecular Properties")
            cols = st.columns(4)
            metrics = [
                ("Molecular Weight", f"{properties['Molecular Weight']} g/mol"),
                ("LogP", properties["LogP"]),
                ("TPSA", f"{properties['TPSA']} Å²"),
                ("Ring Count", properties["Ring Count"]),
                ("H-Bond Donors", properties["Hydrogen Bond Donors"]),
                ("H-Bond Acceptors", properties["Hydrogen Bond Acceptors"]),
                ("Rotatable Bonds", properties["Rotatable Bonds"]),
                ("Fraction Csp3", properties["Fraction Csp3"]),
            ]
            for i, (label, value) in enumerate(metrics):
                with cols[i % 4]:
                    st.metric(label, value)

            st.divider()
            st.subheader("📑 Complete Molecular Property Table")
            report_data = [["Input SMILES", smiles], ["Canonical SMILES", canonical_smiles]]
            report_data.extend([[key, value] for key, value in properties.items()])
            report_df = pd.DataFrame(report_data, columns=["Property", "Value"])
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download Properties Report (CSV)",
                data=report_df.to_csv(index=False).encode("utf-8"),
                file_name="Molecular_Properties_Report.csv",
                mime="text/csv",
            )
            if st.button("📄 Generate Molecular PDF Report"):
                with st.spinner("Generating molecular PDF report..."):
                    pdf = generate_molecule_pdf(smiles, mol, properties)
                st.download_button(
                    "⬇️ Download Molecular Properties PDF",
                    data=pdf,
                    file_name="Molecular_Properties_Report.pdf",
                    mime="application/pdf",
                )

            st.divider()
            st.subheader("🌐 3D Molecular Visualization")
            model_style = st.selectbox(
                "Select 3D Model",
                ["Ball and Stick", "Stick", "Space Filling", "Wireframe", "Cartoon"],
                key="properties_3d_style",
            )
            surface = st.checkbox("Show van der Waals Surface", key="properties_surface")
            if st.button("🚀 Generate 3D Molecular Model", key="properties_generate_3d"):
                with st.spinner("Generating 3D molecular model..."):
                    mol3d = generate_3d_molecule(mol)
                if mol3d is None:
                    st.error("Unable to generate the 3D structure.")
                else:
                    display_3d_structure(mol3d, model_style, surface)


# ============================================================
# STRUCTURE-PROPERTY ANALYSIS
# ============================================================

elif page == "📈 Structure–Property Analysis":
    st.title("📈 Structure–Property Analysis")
    st.write("Compare multiple molecules. Enter one molecule per line as **Molecule Name, SMILES**.")
    default_data = """Ethanol,CCO
Benzene,c1ccccc1
Phenol,Oc1ccccc1
Acetic Acid,CC(=O)O
Aspirin,CC(=O)Oc1ccccc1C(=O)O"""
    molecule_input = st.text_area("Enter Molecules", value=default_data, height=200)
    rows = []
    for line in molecule_input.splitlines():
        if "," not in line:
            continue
        name, smiles = line.split(",", 1)
        mol = get_molecule(smiles.strip())
        if mol:
            prop = calculate_properties(mol)
            rows.append({
                "Molecule": name.strip(),
                "MW": prop["Molecular Weight"],
                "LogP": prop["LogP"],
                "TPSA": prop["TPSA"],
                "HBD": prop["Hydrogen Bond Donors"],
                "HBA": prop["Hydrogen Bond Acceptors"],
                "Rotatable Bonds": prop["Rotatable Bonds"],
                "Ring Count": prop["Ring Count"],
            })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.subheader("📋 Comparison Table")
        st.download_button(
            "⬇️ Download Comparison Table (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="structure_property_comparison.csv",
            mime="text/csv",
            key="download_structure_property_csv",
        )
    else:
        st.warning("Enter at least one valid molecule as: Molecule Name, SMILES")


# ============================================================
# ASSESSMENT
# ============================================================

elif page == "📝 Assessment":
    st.title("📝 Cheminformatics Assessment")
    questions = [
        {"question": "What is the full form of SMILES?",
         "options": ["Simplified Molecular Input Line Entry System",
                     "Standard Molecular Information Language Encoding System",
                     "Simple Molecular Identification and Labeling System"],
         "answer": "Simplified Molecular Input Line Entry System"},
        {"question": "What does a SMILES string represent?",
         "options": ["A molecular structure using text", "A spectroscopy technique",
                     "A laboratory instrument"],
         "answer": "A molecular structure using text"},
        {"question": "Which SMILES represents ethanol?",
         "options": ["CCO", "c1ccccc1", "CC(=O)O"], "answer": "CCO"},
        {"question": "Which SMILES represents benzene?",
         "options": ["c1ccccc1", "CCO", "O=C=O"], "answer": "c1ccccc1"},
        {"question": "In SMILES, parentheses are mainly used to indicate:",
         "options": ["Branches", "Molecular weight", "Atom colour"], "answer": "Branches"},
        {"question": "Which property is commonly associated with molecular lipophilicity?",
         "options": ["LogP", "TPSA", "HBD"], "answer": "LogP"},
        {"question": "What does TPSA describe?",
         "options": ["Topological Polar Surface Area", "Total Pi Surface Area",
                     "Thermal Property Surface Analysis"],
         "answer": "Topological Polar Surface Area"},
        {"question": "What does HBA mean?",
         "options": ["Hydrogen Bond Acceptor", "Heavy Bond Atom", "Hydrogen Bond Analysis"],
         "answer": "Hydrogen Bond Acceptor"},
        {"question": "What does HBD mean?",
         "options": ["Hydrogen Bond Donor", "Heavy Bond Descriptor", "Hydrogen Bond Distance"],
         "answer": "Hydrogen Bond Donor"},
        {"question": "A double bond contains how many pi (π) bonds?",
         "options": ["1", "2", "0"], "answer": "1"},
        {"question": "A triple bond contains how many pi (π) bonds?",
         "options": ["2", "1", "3"], "answer": "2"},
        {"question": "Which model is useful for visualizing approximate molecular volume?",
         "options": ["Space Filling", "Wireframe only", "Text SMILES"],
         "answer": "Space Filling"},
    ]
    student_answers = []
    for i, q in enumerate(questions):
        answer = st.radio(q["question"], q["options"], index=None, key=f"assessment_v2_{i}")
        student_answers.append(answer)
    if st.button("Submit Assessment"):
        if any(answer is None for answer in student_answers):
            st.warning("Please answer all questions before submitting the assessment.")
            st.stop()
        score = sum(student_answers[i] == q["answer"] for i, q in enumerate(questions))
        percentage = score / len(questions) * 100
        st.success(f"Your Score: {score}/{len(questions)}")
        st.metric("Percentage", f"{percentage:.1f}%")
        if percentage >= 80:
            st.balloons()
            st.success("🎉 Excellent performance!")
        elif percentage >= 50:
            st.info("👍 Good! Continue practicing.")
        else:
            st.warning("📚 Review the Theory section and try again.")


# ============================================================
# FINAL PROJECT REPORT
# ============================================================

elif page == "📄 Final Project Report":
    st.title("📄 Final Student Molecular Analysis Report")
    st.write("This report uses the latest valid SMILES entered in the laboratory.")
    name = st.text_input(
        "Student Name for Report",
        value=st.session_state.get("student_name", DEFAULT_STUDENT_NAME),
        key="final_report_name",
    )
    reg = st.text_input(
        "Registration Number for Report",
        value=st.session_state.get("registration_number", DEFAULT_REGISTRATION_NUMBER),
        key="final_report_reg",
    )
    default_smiles = st.session_state.get(
        "latest_smiles", st.session_state.get("active_smiles", "CCO")
    )
    report_smiles = st.text_input(
        "SMILES for Final Report", value=default_smiles, key="final_report_smiles"
    )
    report_mol = get_molecule(report_smiles)
    if report_mol is None:
        st.error("❌ Please enter a valid SMILES notation for the final report.")
    else:
        report_properties = calculate_properties(report_mol)
        st.success("✅ Valid molecule ready for the final report.")
        st.write("**Molecular Formula:**", report_properties["Molecular Formula"])
        st.write("**Molecular Weight:**", report_properties["Molecular Weight"], "g/mol")
        st.write("**Sigma Bonds:**", report_properties["Sigma (σ) Bonds"])
        st.write("**Pi Bonds:**", report_properties["Pi (π) Bonds"])
        if st.button("📄 Generate Personalized Final Project Report"):
            with st.spinner("Generating your personalized report with structures and properties..."):
                pdf = generate_final_report(
                    name, reg, report_smiles, report_mol, report_properties
                )
            st.success("✅ Final Project Report Generated Successfully!")
            st.download_button(
                label="⬇️ Download Personalized Final Project Report PDF",
                data=pdf,
                file_name="Varshan_Cheminformatics_Final_Report.pdf",
                mime="application/pdf",
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "🧪 Cheminformatics Virtual Laboratory | Student: Varshan | "
    "Developed using Python, Streamlit and RDKit"
)
