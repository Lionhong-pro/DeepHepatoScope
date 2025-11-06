#pip install dash scanpy scikit-learn umap-learn numpy pandas plotly pillow seaborn matplotlib scipy gseapy statsmodels tensorflow keras
# import multiprocessing
# multiprocessing.set_start_method("spawn", force=True)

# import anndata as adata
import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES" #Need this for Apple
import pickle
import dash
from dash import dcc, html, Input, Output, State, callback_context, DiskcacheManager, set_props, dash_table
import dash_bootstrap_components as dbc
import diskcache

import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import issparse
from scipy.stats import ttest_ind
# from statsmodels.stats.multitest import multipletests
# from scipy.stats import mannwhitneyu

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import normalize
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import LabelEncoder, normalize
# import umap.umap_ as umap
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import base64
from io import BytesIO
import time
# import time
# import os
# os.environ["OMP_NUM_THREADS"] = "1"  # Limit threads to 1 to reduce noise
# import multiprocessing
# multiprocessing.set_start_method("spawn", force=True)  # Use 'spawn' for Mac if needed
# from PIL import Image
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use("Agg")
import traceback
# import gseapy as gp
# from gseapy import prerank
# from gseapy import gseaplot
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
import plotly.tools as tls
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.preprocessing import normalize
# from sklearn.model_selection import train_test_split
# from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import scaled_dot_product_attention

# from sklearn.preprocessing import LabelEncoder, normalize
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.utils import shuffle
#import anndata
from tqdm import tqdm

import networkx as nx
from collections import Counter
from matplotlib.patches import Wedge, Patch
import community.community_louvain as community_louvain
from matplotlib.patches import Polygon, Circle
from scipy.spatial import ConvexHull
from scipy import sparse
import io
import zipfile

# Create cache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], background_callback_manager=background_callback_manager)
app.title = "DeepHepatoScope"

#Leave the specific order as this works for both scRNA and spatial h5ad files
def encode_anndata(anndata):
    dict = {
            'X': anndata.X.toarray() if hasattr(anndata.X, "toarray") else anndata.X,
            'obs': anndata.obs.to_dict(orient='index'), #orient='index'
            'var_names': anndata.var_names.tolist(),
            'var': anndata.var.to_dict() #orient='index
        }
    return dict

def decode_dict(dict):
    if isinstance(dict['X'], list):
        dict['X'] = np.array(dict['X'])  # Convert list to NumPy array
    elif isinstance(dict['X'], np.ndarray) and dict['X'].ndim == 1:
        dict['X'] = dict['X'].reshape(-1, 1)  # Ensure 2D array
    obs_df = pd.DataFrame.from_dict(dict['obs'], orient='index')
    var_df = pd.DataFrame.from_dict(dict['var'], orient='index')
    if var_df.shape[0] != dict['X'].shape[1]:
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(dict['X'].shape[1])])  # Placeholder
    anndata = sc.AnnData(X=dict['X'], obs=obs_df, var=var_df)
    anndata.var_names = dict['var_names']
    return anndata

# Load the saved image and convert it to base64 for Dash rendering
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

#Assume the genes in the adata are ordered correctly already
def calculate_DEGs(adata_tmp):
    X_tmp = adata_tmp.X
    gene_names_tmp  = adata_tmp.var_names.tolist()
    cell_labels_list = adata_tmp.obs[["Predicted_Class"]].copy()
    print(cell_labels_list)

    nbins = 120
    outlier_ratio = 3.0
    # Store top genes per celltype here
    top_genes_dict = {}

    print("🔬 Calculating distributional log2FC for ALL genes...\n")

    #cell_labels = adata_tmp.obs["Type"] # - use for TD ONLY!
    cell_labels = cell_labels_list["Predicted_Class"] #Added this
    unique_celltypes = np.unique(cell_labels)
    print("unique_celltypes:", unique_celltypes)

    print("X_tmp:", X_tmp)

    for celltype in tqdm(unique_celltypes):
        is_target = (cell_labels == celltype).values
        is_other = ~is_target

        logfc_list = []

        for idx, gene in enumerate(gene_names_tmp):
            expr = X_tmp[:, idx].toarray().flatten() if sparse.issparse(X_tmp) else X_tmp[:, idx].flatten() #Added this

            target_expr = expr[is_target]
            other_expr = expr[is_other]

            mean1 = np.mean(target_expr)
            mean2 = np.mean(other_expr)

            if mean1 > 0 and mean2 > 0:
                logfc = np.log2(mean1 / mean2)
                logfc_list.append((gene, logfc))

        # Filter for log2FC >= 1
        logfc_list = [(g, fc) for g, fc in logfc_list if fc >= 1.0]

        # Sort and pick top 50 genes
        top_genes = sorted(logfc_list, key=lambda x: x[1], reverse=True)[:50]
        top_genes_dict[celltype] = top_genes

        print(f"\n🧬 Top 50 genes for {celltype} (log2FC ≥ 1):\n" + "-" * 60)
        for gene, logfc in top_genes:
            print(f"{gene:<15} log2FC = {logfc:.3f}")

    print("\n✅ Done. Results stored in `top_genes_dict`.")
    # print(top_genes_dict)
    return top_genes_dict
    
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.sqrt_dim = torch.tensor(hidden_dim**0.5, dtype=torch.bfloat16)
        self.Query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Value = nn.Linear(hidden_dim, hidden_dim, bias=False)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, x):
        # x: [batch size, seq len, hidden dim]
        query = self.Query(x)
        # query: [batch size, seq len, hidden dim]
        key = self.Key(x)
        # key: [batch size, hidden dim, seq len]
        value = self.Value(x)
        # value: [batch size, seq len, hidden dim]

        attn_weights = F.softmax(torch.bmm(query, key.permute(0, 2, 1)) / self.sqrt_dim, dim=-1)
        # attn_weights: [batch size, seq len, seq len]

        attn_output = torch.bmm(attn_weights, value)
        # attn_output: [batch size, seq len, hidden dim]

        return attn_output, attn_weights

class TransformerClassifier(nn.Module):
    def __init__(self, hidden_dim, seq_len, num_classes, ffn_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # I trust GPT
        self.W = nn.Parameter(torch.randn(seq_len, 1, hidden_dim))
        self.b = nn.Parameter(torch.zeros(seq_len, hidden_dim))

        # self attention
        self.attn_1 = SelfAttention(hidden_dim)
        self.ffn_1 = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
            # nn.LayerNorm([hidden_dim]), # If architecture is different, remove this line
        )

        # to pool and obtain final output
        self.conv_pool = nn.Conv2d(1, 1, kernel_size=(seq_len, 1))
        # final fc
        self.fc = nn.Linear(hidden_dim, num_classes)
        # self.fc = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.25)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, x):
        # x: [batch size, seq len]
        x = x.unsqueeze(-1)
        x = torch.matmul(x.unsqueeze(-2), self.W)  # -> [batch, seq_len, 1, out_dim]
        x = x.squeeze(-2) + self.b            # -> [batch, seq_len, out_dim]
        x = F.gelu(x)

        # Attention Stuff Below
        attn_x_1, attn_1_weights = self.attn_1(x)
        # attn_x_1: [batch size, seq len, hidden dim]
        attn_x_1 = F.gelu(self.ffn_1(attn_x_1))
        x = attn_x_1

        x = x.unsqueeze(1)
        # x: [batch size, seq len, hidden dim]
        x = F.gelu(self.conv_pool(x))
        # x: [batch size, 16, 1, hidden dim]
        x = x.squeeze()
        # x: [batch size, 128]
        return self.fc(x).float(), attn_1_weights

import requests, pandas as pd, time, random

def enrichr_go(gene_list):
    ENRICHR_URL = "https://maayanlab.cloud/Enrichr"
    """Submit genes to Enrichr and return GO Biological Process enrichment."""
    payload = {"list": "\n".join(gene_list), "description": "batch_test"}
    add_resp = requests.post(f"{ENRICHR_URL}/addList", files=payload)
    user_list_id = add_resp.json()["userListId"]

    params = {"userListId": user_list_id, "backgroundType": "GO_Biological_Process_2023"}
    resp = requests.get(f"{ENRICHR_URL}/enrich", params=params)
    results = resp.json()["GO_Biological_Process_2023"]

    cols = [
        "Rank", "Term", "P-value", "Z-score", "Combined Score",
        "Genes", "Adjusted P-value", "Old P-value", "Old Adjusted P-value"
    ]
    df = pd.DataFrame(results, columns=cols)
    return df[df["Adjusted P-value"] <= 0.05].head(5)

# App layout
app.layout = dbc.Container([
    dcc.Store(id="target-data-store", storage_type="memory"),
    dcc.Store(id="target-data-sub-store", storage_type="memory"),
    dcc.Store(id="annotations-store", storage_type="memory"),
    dcc.Store(id="spatial-plot-store", storage_type="memory"),
    dcc.Store(id="pie-plot-store", storage_type="memory"),
    dcc.Store(id="umap-plot-store", storage_type="memory"),
    dcc.Store(id="tsne-plot-store", storage_type="memory"),
    dcc.Store(id="gcn-plot-store", storage_type="memory"),
    dcc.Store(id="bar-plot-store", storage_type="memory"),
    dcc.Store(id="heatmap-plot-store", storage_type="memory"),
    dcc.Store(id="enrichr-plot-store", storage_type="memory"),
    dcc.Store(id="module-genes-store", storage_type="memory"),
    dcc.Store(id="pathways-store", storage_type="memory"),
    dcc.Download(id="download-store"),
    
    # Header Section
    html.Div([
        html.H1(
            "DeepHepatoScope",
            className="text-center mb-3",
            style={
                #"fontFamily": "'Menlo'", #'SF Mono', 'Menlo', 'Consolas', 'Liberation Mono', monospace"
                "fontWeight": "600",
                "color": "#1e3a8a",
                "fontSize": "2.5rem",
                "letterSpacing": "1px",
                "textShadow": "0 0 6px rgba(30, 58, 138, 0.25)"
            }
        ),
        dbc.Alert([
            html.I(className="bi bi-info-circle-fill me-2"),
            "Thank you for using DeepHepatoScope! If you used DeepHepatoScope in your work, please cite: [CITATION]"
        ], color="white", className="text-center mb-4", style={"border": "none", "boxShadow": "none"}) #0 2px 4px rgba(0,0,0,0.1)
    ], className="mt-4 mb-4"),
    
    # Pre-loaded dataset buttons storage
    dbc.Row([
        dcc.Store(id="selected-button", data=None),
        html.Div(id="advanced-section", style={
            "display": "none", 
            "overflow": "hidden", 
            "height": "0px", 
            "transition": "height 0.5s ease-in-out", 
            "marginTop": "20px"
        }),
        dbc.Col(html.Div(id="card-error-message", className="text-danger mt-2")),
    ], className="mb-4"),

    dcc.Store(id="model-settings-store", storage_type="memory"),

    # Upload Target Dataset Section
    dbc.Card([
        dbc.CardBody([
            html.H4("1. Upload Target Dataset", className="mb-3", style={
                "fontWeight": "600",
                "color": "#1e3a8a"
            }),
            
            dbc.Row([
                dbc.Col(dcc.Upload(
                    id="upload-scrna",
                    children=html.Div([
                        html.I(className="bi bi-cloud-upload", style={"fontSize": "2rem", "color": "#0d6efd"}),
                        html.Br(),
                        html.Span("Drag and Drop or ", style={"fontSize": "0.95rem"}),
                        html.Strong("Click to Upload", style={"color": "#0d6efd"}),
                        html.Br(),
                        html.Span("Target (.h5ad)", style={"fontSize": "0.9rem", "color": "#6c757d"})
                    ]),
                    style={
                        "width": "100%", 
                        "height": "120px",
                        "borderWidth": "2px", 
                        "borderStyle": "dashed", 
                        "borderRadius": "10px",
                        "borderColor": "#0d6efd",
                        "textAlign": "center",
                        "backgroundColor": "#f8f9fa",
                        "cursor": "pointer",
                        "transition": "all 0.3s ease",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center"
                    },
                    multiple=False
                ), width=12, lg=8, className="mb-3 mb-lg-0"),
                dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-file-earmark-text me-2"),
                        "Load CosMx Example"
                    ], id="load-example-cosmx", color="primary", className="w-100 mb-2", style={
                        "borderRadius": "8px",
                        "fontWeight": "500"
                    }),
                    dbc.Button([
                        html.I(className="bi bi-file-earmark-text me-2"),
                        "Load Xenium Example"
                    ], id="load-example-xenium", color="primary", className="w-100", style={
                        "borderRadius": "8px",
                        "fontWeight": "500"
                    })
                ], width=12, lg=4)
            ]),

            dbc.Progress(id="target-progress", striped=True, animated=True, style={
                "margin-top": "20px", 
                "display": "none",
                "height": "20px"
            }),
            html.P(id="target-upload-status", className="text-success mt-2", style={"fontWeight": "500"}),
            # Warning Box
            dbc.Alert([
                html.Div([
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    html.Strong("Important Requirements:")
                ], className="mb-2"),
                html.Ul([
                    html.Li(["DeepHepatoScope runs on ", html.Strong("single cell level"), " RNA count data."]),
                    html.Li(["Provide input file in ", html.Strong(".h5ad"), " format and place it in the ", html.Strong("same directory"), " as app.py."]),
                    html.Li(["For ", html.Strong("spatial transcriptomics"), " datasets, place coordinates in the X_spatial and Y_spatial metadata columns of your .h5ad object"])
                ], className="mb-0", style={"paddingLeft": "20px"})
            ], color="danger", className="mb-3", style={
                "border": "none",
                "borderRadius": "8px",
                "boxShadow": "0 2px 8px rgba(220, 53, 69, 0.2)"
            }),
        ])
    ], className="mb-4", style={"boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "borderRadius": "10px", "border": "none"}),
    
    # Model Selection Section
    dbc.Card([
        dbc.CardBody([
            html.H5("2. Select Model Type", className="mb-3", style={"fontWeight": "600", "color": "#1e3a8a"}),
            dbc.RadioItems(
                id="model-check",
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[
                    {"label": "CosMx SMI (1000 genes)", "value": "cosmx"},
                    {"label": "Xenium hMulti_v1 (474 genes)", "value": "xenium"},
                    {"label": "Custom dataset", "value": "custom"},
                ],
                value="scrna",
                style={"flexWrap": "wrap"}
            ),
        ])
    ], className="mb-4", style={"boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "borderRadius": "10px", "border": "none"}),

    # Custom Model Configuration
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(
                    html.Div([
                        dbc.Label("Number of overlapping genes:", id="custom-num-genes-label", 
                                 html_for="num-genes", className="fw-bold mb-2", 
                                 style={"display": "none", "color": "#495057"}),
                        dbc.Input(
                            id="custom-num-genes",
                            type="text",
                            placeholder="Copy training script output",
                            style={"display": "none", "borderRadius": "8px"}
                        ),
                    ], className="mb-3")
                )
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        dbc.Label("Enter model weights name:", id="custom-model-weights-label", 
                                 html_for="model-weights", className="fw-bold mb-2",
                                 style={"display": "none", "color": "#495057"}),
                        dbc.Input(
                            id="custom-model-weights",
                            type="text",
                            placeholder="model_weights.pt",
                            style={"display": "none", "borderRadius": "8px"}
                        ),
                        dbc.Alert([
                            html.I(className="bi bi-exclamation-triangle-fill me-2"),
                            html.Strong("Note: "),
                            "Place model weights file in the same directory as app.py"
                        ], color="danger", id="custom-weights-warning", className="mt-2", style={
                            "display": "none",
                            "border": "2px solid #dc3545",
                            "borderRadius": "8px",
                            "boxShadow": "0 2px 8px rgba(220, 53, 69, 0.2)",
                            "fontSize": "0.9rem"
                        }),
                    ])
                )
            ]),
        ])
    ], id="custom-model-card", className="mb-4", style={
        "display": "none",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)", 
        "borderRadius": "10px", 
        "border": "none"
    }),

    dcc.Store(id="custom-input-store", storage_type="memory"),
    dcc.Store(id="target-data-settings-store", storage_type="memory"),

    # Advanced Clustering Settings
    dbc.Card([
        dbc.CardBody([
            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col(html.Label("Number of variable genes to use in PCA:", className="fw-bold"), width=12, md=6),
                        dbc.Col(dcc.Input(id="variable_genes-input", value=1000, type="number", min=1, step=1, 
                                         style={"borderRadius": "8px", "width": "100%"}), width=12, md=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(html.Label("Number of PCA dimensions to use in Clustering:", className="fw-bold"), width=12, md=6),
                        dbc.Col(dcc.Input(id="pca_dims-input", value=50, type="number", min=1, step=1,
                                         style={"borderRadius": "8px", "width": "100%"}), width=12, md=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(html.Label("What to calculate?", className="fw-bold"), width=12, md=6),
                        dbc.Col(dcc.Checklist(id="calculation-checklist", 
                                             options=[
                                                 {"label": " UMAP", "value": "umap"}, 
                                                 {"label": " t-SNE", "value": "tsne"}
                                             ],
                                             style={"fontSize": "1rem"}), width=12, md=6)
                    ])
                ], title="Advanced Clustering Settings", style={"borderRadius": "8px"})
            ], start_collapsed=False, style={"border": "none"})
        ])
    ], className="mb-4", style={"boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "borderRadius": "10px", "border": "none"}),
    
    # Train and classify button
    dbc.Card([
    dbc.CardBody([
        dbc.Row([
            # Left Column: Train button + progress
            dbc.Col([
                dbc.Button([
                    html.I(className="bi bi-play-circle-fill me-2"),
                    "3. Analyse and Annotate Target Cell Types"
                ],
                id="train-button",
                color="primary",
                size="lg",
                className="w-100",
                style={
                    "borderRadius": "8px",
                    "fontWeight": "600",
                    "padding": "12px",
                    "boxShadow": "0 4px 8px rgba(25, 135, 84, 0.3)"
                }),
                html.Progress(id="train-progress-bar", value="0", max="100", style={"width": "100%"}, className="mt-3"),
                html.Div(id="train-progress-text"),
                html.Div(id="output", className="mt-3"),
                dcc.Store(id="run-train-trigger", storage_type="memory")
            ], width=12, lg=6),


            dbc.Col([
                dbc.Row([
                dbc.Col([
                dbc.Button([
                    html.I(className="bi bi-download me-2"),
                    "Download annotations as .csv"
                ],
                id="download-annotations",
                color="success",
                size="lg",
                className="w-100 mb-3",
                style={
                    "borderRadius": "8px",
                    "fontWeight": "600",
                    "padding": "12px",
                    "fontSize": "1.2rem",
                    "boxShadow": "0 4px 8px rgba(25, 135, 84, 0.3)",
                    "display": "none"
                })]),

                dbc.Col([
                dbc.Button([
                    html.I(className="bi bi-download me-2"),
                    "Download plots as .png"
                ],
                id="download-analysis-plots",
                color="success",
                size="lg",
                className="w-100 mb-3",
                style={
                    "borderRadius": "8px",
                    "fontWeight": "600",
                    "padding": "12px",
                    "fontSize": "1.2rem",
                    "boxShadow": "0 4px 8px rgba(25, 135, 84, 0.3)",
                    "display": "none"
                })])]),

                html.P(id="gcn-status", className="mt-auto mb-0", style={"color": "black", "fontWeight": "500"})
            ], width=12, lg=6, style={"display": "flex", "flexDirection": "column"})
        ]),

        # Optional status below entire row
        html.P(id="classification-status", className="mt-2 mb-0", style={"color": "black", "fontWeight": "500"})
    ])
], className="mb-4", style={"boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "borderRadius": "10px", "border": "none"}),

    # Visualization stores
    dcc.Store(id="marker-gene-store", storage_type="memory"),
    dcc.Store(id="output-df-store", storage_type="memory"),

dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    # Left column: spatial plot
                    dbc.Col(
                        html.Div(id="spatial-plot-container"),
                        width=6
                    ),

                    # Right column: pie plot
                    dbc.Col(
                        html.Div(id="pie-plot-container"),
                        width=6
                    ),
                ]),

                # Then your next row for UMAP and t-SNE
                dbc.Row([
                    dbc.Col(html.Div(id="umap-plot-container")),
                    dbc.Col(html.Div(id="tsne-plot-container")),
                ]),

                dcc.Graph(id="spatial-plot", style={'display': 'none','height':'600px','width':'600px'}),
                dcc.Graph(id="umap-plot", style={'display': 'none','height':'600px','width':'600px'}),
                dcc.Graph(id="tsne-plot", style={'display': 'none','height':'600px','width':'600px'}),

                dcc.Input(id="marker_gene-input", placeholder="Enter marker gene name", type="text",
                          style={'display':'none','borderRadius':'8px','marginRight':'10px','padding':'10px'}),
                dbc.Button("Show Canonical Marker Genes", id="marker-button", color="secondary",
                           className="mt-2 me-2", style={'display':'none','borderRadius':'8px'}),
                dbc.Button("Return to original plot", id="return-button", color="secondary",
                           className="mt-2", style={'display':'none','borderRadius':'8px'}),
                html.Div(id="marker-gene-status", className="mt-2")
            ], width=12)
        ])
    ])
], className="mb-4", style={"boxShadow":"0 4px 6px rgba(0,0,0,0.1)","borderRadius":"10px","border":"none"}),

    # Visualizdbc.Card([
    dbc.Card([
        dbc.CardBody([
            html.H4("Gene Coexpression Network (GCN) Analysis", className="mb-4", style={
                "fontWeight": "600", "color": "#1e3a8a"
            }),
            dbc.Row([
                # Left Column: GCN Analysis button, progress, outputs, alert
                dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-play-circle-fill me-2"),
                        "4. Perform Gene Coexpression Network (GCN) Analysis"
                    ],
                    id="network-button",
                    color="primary",
                    size="lg",
                    className="w-100",
                    style={
                        "borderRadius": "8px",
                        "fontWeight": "600",
                        "padding": "12px",
                        "boxShadow": "0 4px 8px rgba(13, 110, 253, 0.3)",
                    }),
                    html.Progress(id="gcn-progress-bar", value="0", max="100",
                                style={"width": "100%"}, className="mt-3"),
                    html.Div(id="gcn-progress-text"),
                    html.Div(id="output", className="mt-3"),
                    dcc.Store(id="run-network-trigger", storage_type="memory"),
                    dbc.Alert([
                        html.I(className="bi bi-info-circle-fill me-2"),
                        html.Strong("Important: "),
                        "Only press once model inference is complete."
                    ], color="warning", className="mb-2 mt-3", style={
                        "border": "2px solid #ffc107",
                        "borderRadius": "8px",
                        "fontSize": "0.9rem"
                    })
                ], width=12, lg=6),

                # Right Column: Download button (top) + gcn-status (bottom)
                dbc.Col([
                dbc.Row([
                    dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-download me-2"),
                        "Download module genes and pathways as .csv"
                    ],
                    id="download-module-genes",
                    color="success",
                    size="lg",
                    className="w-100 mb-3",
                    style={
                        "borderRadius": "8px",
                        "fontWeight": "600",
                        "padding": "12px",
                        "fontSize": "1.2rem",
                        "boxShadow": "0 4px 8px rgba(25, 135, 84, 0.3)",
                        "display": "none"
                    })
                    ]),
                    dbc.Col([
                    dbc.Button([
                        html.I(className="bi bi-download me-2"),
                        "Download plots as .png"
                    ],
                    id="download-gcn-plots",
                    color="success",
                    size="lg",
                    className="w-100 mb-3",
                    style={
                        "borderRadius": "8px",
                        "fontWeight": "600",
                        "padding": "12px",
                        "fontSize": "1.2rem",
                        "boxShadow": "0 4px 8px rgba(25, 135, 84, 0.3)",
                        "display": "none"
                    })
                    ]),
                    html.P(id="gcn-status", className="mt-auto mb-0",
                        style={"color": "black", "fontWeight": "500"})])
                ], width=12, lg=6, style={"display": "flex", "flexDirection": "column"})
            ])
        ])
    ], className="mb-4", style={
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
        "borderRadius": "10px",
        "border": "none"
    }),

    # GCN Results Section
    dbc.Card([
        dbc.CardBody([
            html.H4("GCN Analysis Results", className="mb-4", style={"fontWeight": "600", "color": "#1e3a8a"}),
            html.Div([
                html.Div(id="gcn-plot-container"),
                html.Div(id="bar-plot-container"),
                html.Div(id="heatmap-plot-container"),
                html.Div(id="enrichr-plot-container")
            ], style={
                "width": "60%",
                "margin": "0 auto",
                "display": "flex",
                "flexDirection": "column",
                "gap": "10px",            # 👈 adds 10 px space between each plot
                "textAlign": "center"
            }),
            html.Div(id="cluster-table-container", children=[
                html.H4("Module Genes"),
                # html.Button("Download CSV", id="download-clusters-btn", style={"display": "none"}),
                # dcc.Download(id="download-clusters-csv"),
                # html.Div("Run analysis to see results...", id="cluster-placeholder")
            ]),
        ])
    ], className="mb-4", style={"boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "borderRadius": "10px", "border": "none"}),
], fluid=True, style={"maxWidth": "14000px", "backgroundColor": "#e0f2f7", "padding": "20px"}) #f8f9fa

def update_progress(value, text):
    set_props("train-progress-bar", {"value": value})
    time.sleep(0.5)  # Ensure update is processed
    set_props("train-progress-text", {"children": text})
    time.sleep(0.5)

def update_progress_gcn(value, text):
    set_props("gcn-progress-bar", {"value": value})
    time.sleep(0.5)  # Ensure update is processed
    set_props("gcn-progress-text", {"children": text})
    time.sleep(0.5)

def update_spatial_plot(plot):
    set_props("spatial-plot-container", {"children": plot})
    time.sleep(0.5)  # Ensure update is processed

def update_pie_plot(plot):
    set_props("pie-plot-container", {"children": plot})
    time.sleep(0.5)  # Ensure update is processed

def update_umap_plot(plot):
    set_props("umap-plot-container", {"children": plot})
    time.sleep(0.5)  # Ensure update is processed

def update_tsne_plot(plot):
    set_props("tsne-plot-container", {"children": plot})
    time.sleep(0.5)  # Ensure update is processed

def update_gcn_plot(plot):
    set_props("gcn-plot-container", {"children": plot})
    time.sleep(0.5)  # Ensure update is processed

def update_bar_plot(plot):
    set_props("bar-plot-container", {"children": plot})
    time.sleep(0.5)  # Ensure update is processed

def update_heatmap_plot(plot):
    set_props("heatmap-plot-container", {"children": plot})
    time.sleep(0.5)  # Ensure update is processed

def update_enrichr_plot(plot):
    set_props("enrichr-plot-container", {"children": plot})
    time.sleep(0.5)  # Ensure update is processed

    #     dcc.Store(id="annotations-store", storage_type="memory"),
    # dcc.Store(id="spatial-plot-store", storage_type="memory"),
    # dcc.Store(id="umap-plot-store", storage_type="memory"),
    # dcc.Store(id="tsne-plot-store", storage_type="memory"),
    # dcc.Store(id="gcn-plot-store", storage_type="memory"),
    # dcc.Store(id="bar-plot-store", storage_type="memory"),
    # dcc.Store(id="heatmap-plot-store", storage_type="memory"),
    # dcc.Store(id="module-genes-store", storage_type="memory"),

@app.callback(
    Output("download-store", "data", allow_duplicate=True),
    Input("download-annotations", "n_clicks"),
    State("annotations-store", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, data):
    if not data:
        return dash.no_update
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "annotations.csv", index=False)

@app.callback(
    Output("download-store", "data", allow_duplicate=True),
    Input("download-analysis-plots", "n_clicks"),
    State("spatial-plot-store", "data"),
    State("pie-plot-store", "data"),
    State("umap-plot-store", "data"),
    State("tsne-plot-store", "data"),
    prevent_initial_call=True
)
def download_plots(n_clicks, spatial_data, pie_data, umap_data, tsne_data):
    if not any([spatial_data, pie_data, umap_data, tsne_data]):
        return dash.no_update

    # Helper: Decode base64 image
    def decode_image(data_url):
        if not data_url:
            return None
        header, encoded = data_url.split(",", 1)
        return base64.b64decode(encoded)

    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        if spatial_data:
            zf.writestr("spatial_plot.png", decode_image(spatial_data))
        if pie_data:
            zf.writestr("celltype_composition_pie_plot.png", decode_image(pie_data))
        if umap_data:
            zf.writestr("umap_plot.png", decode_image(umap_data))
        if tsne_data:
            zf.writestr("tsne_plot.png", decode_image(tsne_data))

    zip_buffer.seek(0)
    return dcc.send_bytes(zip_buffer.getvalue(), "analysis_plots.zip")

@app.callback(
    Output("download-store", "data", allow_duplicate=True),
    Input("download-gcn-plots", "n_clicks"),
    State("gcn-plot-store", "data"),
    State("bar-plot-store", "data"),
    State("heatmap-plot-store", "data"),
    State("enrichr-plot-store", "data"),
    prevent_initial_call=True
)
def download_plots(n_clicks, gcn_data, bar_data, heatmap_data, enrichr_data):
    if not any([gcn_data, bar_data, heatmap_data]):
        return dash.no_update

    # Helper: Decode base64 image
    def decode_image(data_url):
        if not data_url:
            return None
        header, encoded = data_url.split(",", 1)
        return base64.b64decode(encoded)

    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        if gcn_data:
            zf.writestr("gcn_plot.png", decode_image(gcn_data))
        if bar_data:
            zf.writestr("bar_plot.png", decode_image(bar_data))
        if heatmap_data:
            zf.writestr("heatmap_plot.png", decode_image(heatmap_data))
        if enrichr_data:
            zf.writestr("enrichr_plot.png", decode_image(enrichr_data))


    zip_buffer.seek(0)
    return dcc.send_bytes(zip_buffer.getvalue(), "gcn_plots.zip")

@app.callback(
    Output("download-store", "data", allow_duplicate=True),
    Input("download-module-genes", "n_clicks"),
    State("module-genes-store", "data"),
    State("pathways-store", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, data1, data2):
    if not data1:
        return dash.no_update
    if not data2:
        return dash.no_update
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    # Create an in-memory zip
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write both CSVs to the zip file
        zf.writestr("module_genes.csv", df1.to_csv(index=False))
        zf.writestr("enrichr_pathways.csv", df2.to_csv(index=False))
    buffer.seek(0)
    return dcc.send_bytes(buffer.getvalue(), "module_genes_and_pathways.zip")

# Callbacks
#upload scrna data
@app.callback(
    Output("target-upload-status", "children", allow_duplicate = True),
    Output("upload-scrna", "children", allow_duplicate=True),
    Output("upload-scrna", "style", allow_duplicate=True),
    Output("upload-scrna", "disabled", allow_duplicate=True),
    Output("target-data-store", "data", allow_duplicate=True),
    Output("model-check", "value", allow_duplicate=True),
    Output("model-check", "style", allow_duplicate=True),
    Input("load-example-cosmx", "n_clicks"),
    prevent_initial_call = True,
)
def upload_target(n_clicks):
    print("Loading CosMx example...")
    import os
    # target_data = sc.read_h5ad("CosMx_demo_2775cells_1000genes.h5ad") #spatial_data_2
    # target_data_dict = encode_anndata(target_data)
    filename = os.path.join(os.path.dirname(__file__), "example_data", "CosMx_demo_2775cells_1000genes.h5ad")
    model_check_disabled_style = {
        "opacity": "0.6",
        "pointerEvents": "none",
        "filter": "grayscale(100%)",
        "flexWrap": "wrap"
    }
    return f"Uploaded Target File: CosMx_demo_2775cells_1000genes.h5ad", "Dataset uploaded successfully!", {
        "width": "100%",
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px",
        "backgroundColor": "green",  # Change background to green
        "color": "darkgreen"
    }, True, {"filename": filename}, "cosmx", model_check_disabled_style #{"display": "none"} # Disable the upload box"

@app.callback(
    Output("target-upload-status", "children", allow_duplicate = True),
    Output("upload-scrna", "children", allow_duplicate=True),
    Output("upload-scrna", "style", allow_duplicate=True),
    Output("upload-scrna", "disabled", allow_duplicate=True),
    Output("target-data-store", "data", allow_duplicate=True),
    Output("model-check", "value", allow_duplicate=True),
    Output("model-check", "style", allow_duplicate=True),
    Input("load-example-xenium", "n_clicks"),
    prevent_initial_call = True,
)
def upload_target(n_clicks):
    print("Loading Xenium example...")
    import os
    # target_data = sc.read_h5ad("Xenium_demo_16833cells_474genes.h5ad") #spatial_data_2
    # target_data_dict = encode_anndata(target_data)
    filename = os.path.join(os.path.dirname(__file__), "example_data", "Xenium_demo_16833cells_474genes.h5ad")
    model_check_disabled_style = {
        "opacity": "0.6",
        "pointerEvents": "none",
        "filter": "grayscale(100%)",
        "flexWrap": "wrap"
    }
    return f"Uploaded Target File: Xenium_demo_16833cells_474genes.h5ad", "Dataset uploaded successfully!", {
        "width": "100%",
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px",
        "backgroundColor": "green",  # Change background to green
        "color": "darkgreen"
    }, True, {"filename": filename}, "xenium", model_check_disabled_style #{"display": "none"} # Disable the upload box"


@app.callback(
    Output("target-upload-status", "children", allow_duplicate = True),
    Output("upload-scrna", "children", allow_duplicate=True),
    Output("upload-scrna", "style", allow_duplicate=True),
    Output("upload-scrna", "disabled", allow_duplicate=True),
    Output("target-data-store", "data", allow_duplicate=True),
    Input("upload-scrna", "contents"),
    State("upload-scrna", "filename"), #file must be in the same directory as app.py
    prevent_initial_call = True,
)
def upload_target(contents, filename):
    # if contents:
    try:
        print("Uploading file...")
        # target_data = sc.read_h5ad(filename)
        # target_data_dict = encode_anndata(target_data) - ENCODING LARGE FILES CRASHES!
        print("File encoded. Returning...")
        return f"Uploaded Target File: {filename}", "Dataset uploaded successfully!", {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "solid",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
            "backgroundColor": "green",  # Change background to green
            "color": "darkgreen"
        }, True, {"filename": filename} # Disable the upload box"
    except Exception as e:
        return "", f"Error reading file: {e}", None, False, None
    # return "No target file uploaded."

@app.callback(
    Output("custom-num-genes-label", "style", allow_duplicate = True),
    Output("custom-num-genes", "style", allow_duplicate = True),
    Output("custom-model-weights-label", "style", allow_duplicate = True),
    Output("custom-model-weights", "style", allow_duplicate = True),
    Input("model-check", "value"),
    prevent_initial_call = True,
)
def upload_custom(model):
    if model=="custom":
        return({"display": "inline-block"}, {"display": "inline-block"}, {"display": "inline-block"}, {"display": "inline-block"})
    else:
        return({"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"})

@app.callback(
    Output("custom-input-store", "data"),
    Input("custom-num-genes", "value"),
    Input("custom-model-weights", "value"), #Must be an input as well, as the callback will only run when the input is triggered; the state just comes along upon triggering
)
def store_inputs(custom_num_genes, custom_model_weights):
    return {"custom_num_genes": custom_num_genes, "custom_model_weights": custom_model_weights}

@app.callback(
    Output("target-data-settings-store", "data"),
    [Input("variable_genes-input", "value"),
     Input("pca_dims-input", "value"),
     #Input("normalisation-dropdown", "value"),
     Input("calculation-checklist", "value")]
)
def update_model_settings(variable_genes, pca_dims, calculation):
    # Create dictionary to store the settings
    return {
        "variable_genes": variable_genes,
        "pca_dims": pca_dims,
        #"normalisation": normalisation,
        "calculation": calculation
    }

# @app.callback(
#     Output("marker-gene-store", "data"),
#     Input("marker_gene-input", "value")
# )
# def update_model_settings(marker_gene):
#     # Create dictionary to store the settings
#     return {
#         "marker_gene": marker_gene,
#     }

@app.callback(
    Output("network-button","children", allow_duplicate=True),
    Output("network-button", "disabled"),
    Output("run-network-trigger", "data"),
    Input("network-button", "n_clicks"),
    prevent_initial_call=True
)
def start_analysis_ui(n_clicks):
    # Immediately change button to spinner state
    spinner_content = html.Span([
        dbc.Spinner(size="sm", color="light"), #, className="me-2"
        "Analysing..."
    ])
    return spinner_content, True, {"start": True}

@app.callback(
    Output("gcn-status", "children"),
    Output("gcn-status", "style"),
    Output("gcn-plot-container", "children"),
    Output("bar-plot-container", "children"),
    Output("heatmap-plot-container", "children"),
    Output("enrichr-plot-container", "children"),
    Output("cluster-table-container", "children"),

    Output("gcn-plot-store", "data", allow_duplicate=True),
    Output("bar-plot-store", "data", allow_duplicate=True),
    Output("heatmap-plot-store", "data", allow_duplicate=True),
    Output("enrichr-plot-store", "data", allow_duplicate=True),
    Output("module-genes-store", "data", allow_duplicate=True),
    Output("pathways-store", "data", allow_duplicate=True),

    Output("download-module-genes", "style"),
    Output("download-gcn-plots", "style"),

    Output("network-button","children", allow_duplicate=True),
    Input("run-network-trigger", "data"),
    State("model-check", "value"),
    State("target-data-store", "data"),
    State("target-data-sub-store", "data"),
    State("output-df-store", "data"),
    background=True,
    running=[
        (Output("gcn-progress-bar", "value"), "0", "100"),
        (Output("gcn-progress-text", "children"), "Starting analysis...", ""),
        (Output("gcn-plot-container", "children"), None, None),
        (Output("bar-plot-container", "children"), None, None),
        (Output("heatmap-plot-container", "children"), None, None)
    ],
    prevent_initial_call=True
)
def gcn_analysis(n_clicks, model_check, target_data_store, target_data_sub_store, output_df_store):
    try:
        print("GCN pipeline started. Importing modules...")
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import networkx as nx
        from collections import Counter
        from matplotlib.patches import Wedge, Patch
        import community.community_louvain as community_louvain
        from matplotlib.patches import Polygon, Circle
        from scipy.spatial import ConvexHull
        import io
        update_progress_gcn("0", "Loading trained model attention weights...")
        print("Model selected:", model_check)
        if model_check=="cosmx":
            attn_matrix = np.load(os.path.join(os.path.dirname(__file__), "attention_weights", "CosMx_850genes_attentionweights_300epochs.npz"))["CosMx_attnweights"]
            seurat_genes = pd.read_csv(os.path.join(os.path.dirname(__file__), "gene_lists", "final_CosMx850_genes.csv"), header=None).iloc[:, 0].astype(str).tolist()
        elif model_check=="xenium":
            attn_matrix = np.load(os.path.join(os.path.dirname(__file__), "attention_weights", "Xenium_360genes_attentionweights_130epochs.npz"))["Xenium_attnweights"]
            seurat_genes = pd.read_csv(os.path.join(os.path.dirname(__file__), "gene_lists", "final_Xenium360_genes.csv"),header=None).iloc[:, 0].astype(str).tolist()

        genes = seurat_genes[:]
        update_progress_gcn("10", "Loading data files...")

        adata = sc.read_h5ad(target_data_store["filename"])
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        target_data_full = adata
        # print("Running scaler fitting...")

        seurat_genes_idx = pd.Index(seurat_genes)
        shared_genes = seurat_genes_idx.intersection(adata.var_names)
        adata = adata[:, shared_genes].copy()

        predicted_labels_list = output_df_store["predicted_labels_list"]
        predicted_labels = np.array(predicted_labels_list)
        adata.obs["Predicted_Class"] = predicted_labels

        update_progress_gcn("20", "Calculating DEGs...")

        temp_DEGs = calculate_DEGs(adata)
        DEGs = {
            cell_type: {gene: round(float(score), 3) for gene, score in genes}
            for cell_type, genes in temp_DEGs.items()
        }

        # Your cell type mapping
        cell_types = [
            "B cells",
            "CAFs",
            "Cholangiocytes",
            "Dendritic cells",
            "GMPs",
            "Hepatocytes",
            "Malignant cells",
            "Mast cells",
            "Monocytes",
            "Myofibroblasts",
            "NK cells",
            "Neutrophils",
            "Pericytes",
            "Plasma cells",
            "Plasmacytoid dendritic cells",
            "Smooth muscle cells",
            "Stellate cells",
            "T cells",
            "TAMs",
            "TECs"
        ]

        update_progress_gcn("30", "Processing attention matrix...")

        # Create a dict of DataFrames for easy labeling
        attn_matrix_labeled = {
            cell_types[i]: pd.DataFrame(attn_matrix[i], index=genes, columns=genes)
            for i in range(len(cell_types))
        }
        # Example usage:
        value = attn_matrix_labeled["B cells"].loc["ACTA2", "ACTA2"]
        print(value)
        print(attn_matrix_labeled.items())
        print("attn_matrix shape:", attn_matrix.shape)
        print("Number of cell types:", len(cell_types))
        print("attn_matrix_labeled keys:", list(attn_matrix_labeled.keys()))

        top20_values_per_celltype = {}

        for celltype, matrix in attn_matrix_labeled.items():
            print(celltype)
            # Get DEGs for this celltype
            deg_genes = list(DEGs.get(celltype, {}).keys())

            # Keep only genes present in the matrix
            deg_genes = [g for g in deg_genes if g in matrix.index]

            if len(deg_genes) < 2:
                print(f"Skipping {celltype} — only {len(deg_genes)} matching DEGs found.")
                continue

            # Subset matrix to only these DEGs in both rows and columns
            sub_matrix = matrix.loc[deg_genes, deg_genes]

            # Stack to long format to get gene1, gene2, and value in one series
            stacked = sub_matrix.stack()

            # Exclude self-edges where gene1 == gene2
            stacked = stacked[stacked.index.get_level_values(0) != stacked.index.get_level_values(1)]

            stacked = stacked[
            (stacked.index.get_level_values(0) != "SPOCK2") &
            (stacked.index.get_level_values(1) != "SPOCK2")
            ]

            # Sort descending by value
            top20 = stacked.sort_values(ascending=False).head(25)

            # Print nicely
            print(f"Top 20 highest attention values for {celltype} (excluding self-edges):")
            for (gene1, gene2), val in top20.items():
                print(f"{gene1} -> {gene2}: {val}")
            print("-" * 40)

            # Store in dict: each value is a DataFrame with columns ['Gene1', 'Gene2', 'Value']
            top20_values_per_celltype[celltype] = (
                top20.reset_index()
                    .rename(columns={0: 'Value', 'level_0': 'Gene1', 'level_1': 'Gene2'})
            )

        #DO NOT INDENT THE PART BELOW! ELSE THE PLOT WILL BE TRUNCATED.
        # ----------------------------
        # Assumes `top20_values_per_celltype` is a dict: celltype -> pandas.DataFrame with columns Gene1,Gene2,Value
        # ----------------------------

        # -------- Build graph & per-edge contributions ----------
        print("Commencing graph generation...")
        update_progress_gcn("40", "Commencing graph generation...")
        G = nx.Graph()
        for celltype, df in top20_values_per_celltype.items():
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                g1, g2, val = str(row['Gene1']), str(row['Gene2']), float(row['Value'])
                if G.has_edge(g1, g2):
                    G[g1][g2]['weight'] += val
                    G[g1][g2]['contrib'][celltype] = G[g1][g2]['contrib'].get(celltype, 0.0) + val
                else:
                    G.add_edge(g1, g2, weight=val, contrib={celltype: float(val)})

        # Compute dominant celltype and "length" attribute
        for u, v, d in G.edges(data=True):
            w = d.get('weight', 0.0)
            d['length'] = 1.0 / w if w > 0 else 1e6
            contrib = d.get('contrib', {})
            d['dominant'] = max(contrib.items(), key=lambda kv: kv[1])[0] if contrib else None

        # -------- Colors ----------
        predefined_colors = {
            "Malignant cells": "#aec7e8","Stellate cells": "#1f77b4","TAMs": "#ff7f0e",
            "T cells": "#ffbb78","TECs": "#2ca02c","B cells": "#98df8a","Erythroid cells": "#d62728",
            "Hepatocytes": "#ff9896","NK cells": "#9467bd","Cholangiocytes": "#c5b0d5","Unclassified": "#8c564b",
            "Mast cells": "#e377c2","Dendritic cells": "#7f7f7f","CAFs": "#bcbd22","Myofibroblasts": "#17becf",
            "Monocytes": "#c49c94","GMPs": "#f7b6d2","Pericytes": "#8c6d31","Smooth muscle cells": "#393b79",
            "Neutrophils": "#9edae5","Plasma cells": "#c7c7c7","Plasmacytoid dendritic cells": "#636363","Others": "#8c8c8c"
        }
        celltypes = sorted(top20_values_per_celltype.keys())
        celltype_colors = {ct: predefined_colors.get(ct, "#999999") for ct in celltypes}

        # -------- Compute node degrees & sizes ----------
        degrees = dict(G.degree())
        deg_values = np.array(list(degrees.values()))
        min_node_size, max_node_size = 500, 3000
        deg_min, deg_max = (deg_values.min(), deg_values.max()) if deg_values.size > 0 else (0,1)

        def scale_node_size(deg):
            if deg_max == deg_min:
                return (min_node_size + max_node_size) / 2
            return min_node_size + (deg - deg_min) / (deg_max - deg_min) * (max_node_size - min_node_size)

        node_sizes = {node: scale_node_size(deg)*5 for node, deg in degrees.items()}
        node_radius_base = 0.0045
        node_radius = {node: node_radius_base * np.sqrt(size / min_node_size) for node, size in node_sizes.items()}

        # -------- Helper to draw pies ----------
        def draw_pie(ax, center, counts, colors, radius=0.04, start_angle=90):
            total = sum(counts)
            if total <= 0:
                ax.add_patch(plt.Circle(center, radius, color='#DDDDDD', ec='k', lw=0.3))
                # return - will return the whole callback function as NoneType. Maybe didn't crash the smaller demos as this part was not called
            angle = start_angle
            for cnt, col in zip(counts, colors):
                if cnt <= 0: continue
                theta2 = angle - (cnt / total) * 360.0
                ax.add_patch(Wedge(center, radius, theta2, angle, facecolor=col, edgecolor='k', lw=0.3))
                angle = theta2

        # -------- Compute per-node edge counts by celltype ----------
        node_edge_ct_counts = {}
        for node in G.nodes():
            ct_counter = Counter()
            for nbr in G.neighbors(node):
                dom = G[node][nbr].get('dominant')
                if dom: ct_counter[dom] += 1
            node_edge_ct_counts[node] = ct_counter

        # -------- Layout per connected component & arrange in triangular formation ----------
        components = list(nx.connected_components(G))
        pos = {}
        gap = 2.0  # spacing between clusters

        # triangular positions: top-left, top-right, bottom-center
        tri_offsets = []
        n = len(components)
        if n == 1:
            tri_offsets = [(0,0)]
        elif n == 2:
            tri_offsets = [(0,0), (3,0)]
        elif n == 3:
            tri_offsets = [(0,0), (2,0), (0.95,-0.5)]
        else:
            # fallback: place in row
            tri_offsets = [(i*gap, 0) for i in range(n)]

        # print(len(components), components)

        # ---- General layout for any number of components ----
        pos = {}
        for comp in components:
            subG = G.subgraph(comp)
            sub_pos = nx.spring_layout(subG, weight='length', k=15, iterations=1000, seed=42)
            pos.update(sub_pos)

        # ---- Plot combined network ----
        fig, ax = plt.subplots(figsize=(25, 25))
        ax.set_aspect('equal')

        # Edge attributes
        edge_colors = [celltype_colors.get(d.get('dominant'), "#999999") for _, _, d in G.edges(data=True)]
        edge_widths = [1.0] * G.number_of_edges()

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7, ax=ax)

        # Draw pie charts for nodes
        for node, (x, y) in pos.items():
            counts = [node_edge_ct_counts[node].get(ct, 0) for ct in celltypes]
            cols = [celltype_colors[ct] for ct in celltypes]
            draw_pie(ax, (x, y), counts, cols, radius=node_radius[node])

        # Add labels
        for node, (x, y) in pos.items():
            ax.text(x, y - node_radius[node] - 0.002, str(node),
                    fontsize=8, ha='center', va='top', color='black')

        # Legend
        legend_handles = [Patch(color=celltype_colors[ct], label=ct) for ct in celltypes]
        ax.legend(handles=legend_handles, title="Cell Types", loc='upper right', bbox_to_anchor=(1.15, 1))

        ax.axis('off')
        plt.title("Gene-Gene Coexpression Network — All Components")
        plt.tight_layout()
        # plt.show()
        
        #DO NOT INDENT THE PART BELOW, ELSE WILL RETURN NONETYPE (?)
        # --- Parameters ---
        seed = 1732349452
        res = 1.0649
        min_nodes = 5   # clusters smaller than this will be merged

        print(f"Random seed: {seed}, Resolution: {res}")

        # --- Run Louvain ---
        partition = community_louvain.best_partition(G, weight='weight', resolution=res, random_state=seed)

        # --- Aggressive small cluster merging ---
        def merge_small_clusters(node_cluster, pos, min_nodes=5):
            merged = True
            while merged:
                merged = False
                cluster_counts = Counter(node_cluster.values())
                small_clusters = {cl for cl, sz in cluster_counts.items() if sz < min_nodes}
                large_clusters = {cl for cl, sz in cluster_counts.items() if sz >= min_nodes}
                if not small_clusters:
                    break

                for sc in small_clusters:
                    sc_nodes = [n for n, cl in node_cluster.items() if cl == sc]
                    if not large_clusters:
                        continue
                    # Compute centroids of large clusters
                    large_centroids = {lc: np.mean([pos[n] for n, cl in node_cluster.items() if cl == lc], axis=0)
                                    for lc in large_clusters}
                    # Centroid of small cluster
                    sc_centroid = np.mean([pos[n] for n in sc_nodes], axis=0)
                    # Nearest large cluster
                    nearest_large = min(large_centroids, key=lambda lc: np.linalg.norm(sc_centroid - large_centroids[lc]))
                    for n in sc_nodes:
                        node_cluster[n] = nearest_large
                    merged = True
            return node_cluster

        node_cluster = merge_small_clusters(partition.copy(), pos, min_nodes=min_nodes)
        cluster_labels = sorted(set(node_cluster.values()))
        n_clusters = len(cluster_labels)
        print(f"Aggressive merging → {n_clusters} communities")

        # --- Renumber clusters from 0 to n-1 ---
        unique_clusters = sorted(set(node_cluster.values()))
        cluster_map = {old: new for new, old in enumerate(unique_clusters)}
        node_cluster = {n: cluster_map[cl] for n, cl in node_cluster.items()}

        # Update cluster_labels and n_clusters
        cluster_labels = sorted(set(node_cluster.values()))
        n_clusters = len(cluster_labels)
        print(f"Final renumbered clusters → {n_clusters} communities")

        # --- Modularity after merging ---
        mod = community_louvain.modularity(node_cluster, G, weight='weight')
        print(f"Modularity after merging: {mod:.4f}")

        print(" ")
        print("Calculating GCN...")
        update_progress_gcn("40", "Calculating GCN...")

        # --- Cluster colors (15 visually distinct) ---
        n_colors = 15
        # --- Define fixed colors for clusters ---
        tab20_colors = plt.get_cmap("tab20").colors
        selected_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18]
        selected_colors = [tab20_colors[i] for i in selected_indices]

        # Pick indices spaced apart for max contrast (darker / vivid ones)
        # selected_indices = [0,2,4,6,8,10,12,14,16,17, 18]  # these are well-separated, not too light

        extra_palettes = [
            plt.get_cmap("Set3").colors,
            plt.get_cmap("Dark2").colors,
            plt.get_cmap("Paired").colors,
        ]

        # Combine selected vivid colors + extra palettes
        all_colors = list(selected_colors)
        for palette in extra_palettes:
            all_colors.extend(palette)

        # Assign to your clusters
        # cluster_colors = {cl: tab20_colors[i % len(tab20_colors)] for cl, i in zip(cluster_labels, selected_indices)}
        cluster_colors = {
            cl: all_colors[i % len(all_colors)]
            for i, cl in enumerate(cluster_labels)
        }

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_aspect('equal')

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7, ax=ax)

        # Draw cluster polygons
        for cl in cluster_labels:
            cl_nodes = [n for n in G.nodes() if node_cluster[n] == cl]
            pts = np.array([pos[n] for n in cl_nodes])
            color = cluster_colors[cl]
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    poly_pts = pts[hull.vertices]
                    ax.add_patch(Polygon(poly_pts, closed=True, facecolor=color, edgecolor=None, alpha=0.12, zorder=0))
                except:
                    ax.scatter(pts[:,0], pts[:,1], s=200, color=color, alpha=0.08, zorder=0)
            elif len(pts) > 0:
                centroid = pts.mean(axis=0)
                r = 0.06 if pts.shape[0]==1 else np.mean(np.linalg.norm(pts - centroid, axis=1))*1.3 + 0.02
                ax.add_patch(Circle(centroid, r, facecolor=color, edgecolor=None, alpha=0.25, zorder=0))

        # Draw pies for nodes
        for node, (x, y) in pos.items():
            counts = [node_edge_ct_counts[node].get(ct,0) for ct in celltypes]
            cols = [celltype_colors[ct] for ct in celltypes]
            draw_pie(ax, (x, y), counts, cols, radius=node_radius[node])

        # Node outlines
        for node, (x, y) in pos.items():
            ax.add_patch(Circle((x, y), node_radius[node], facecolor='none', edgecolor='k', lw=0.25, zorder=3))

        # Node labels
        for node, (x, y) in pos.items():
            ax.text(x, y - node_radius[node]-0.003, str(node), fontsize=6, ha='center', va='top', color='black', zorder=4)

        # Legends
        cluster_handles = [plt.matplotlib.patches.Patch(color=cluster_colors[c], label=f"Community {c}") for c in cluster_labels]
        ct_handles = [plt.matplotlib.patches.Patch(color=celltype_colors[ct], label=ct) for ct in celltypes]
        leg1 = ax.legend(handles=cluster_handles, title="Louvain Communities", loc='upper right', bbox_to_anchor=(1.15,1))
        leg2 = ax.legend(handles=ct_handles, title="Cell Types", loc='lower right', bbox_to_anchor=(1.15,0.2))
        ax.add_artist(leg1)

        ax.axis('off')
        plt.title(f"Gene-Gene Coexpression Network — Louvain Communities\nResolution {res} (n={n_clusters})")
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(os.path.dirname(__file__), "DeepHepatoScope_results", "gcn_plot.png"), format="png", dpi=300, bbox_inches="tight")

        # Save to in-memory buffer instead of file
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        buffer.seek(0)
        plt.close()

        # Encode as base64 for HTML rendering
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        image_src = f"data:image/png;base64,{encoded_image}"
        store_gcn_image = image_src
        gcn_plot_image = html.Img(src=image_src, style={
            "maxWidth": "100%",
            "height": "auto",
            "borderRadius": "10px",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.15)"
        })

        update_gcn_plot(gcn_plot_image)
        print("--- GCN plot saved to DeepHepatoScope_results/gcn_plot.png ---")

        #BAR-PLOT

        print(" ")
        print("Calculating cluster compositions...")
        update_progress_gcn("60", "Calculating cluster compositions...")

        # Convert node_edge_ct_counts to DataFrame
        df_counts = pd.DataFrame.from_dict(node_edge_ct_counts, orient='index').fillna(0)

        # Add cluster info
        df_counts['cluster'] = [node_cluster[n] for n in df_counts.index]

        # Aggregate counts per cluster (sum)
        cluster_sum = df_counts.groupby('cluster').sum()

        # Convert to fractions (each row sums to 1)
        cluster_frac_df = cluster_sum.div(cluster_sum.sum(axis=1), axis=0)

        # Now you can filter used cell types
        used_celltypes = [ct for ct in cluster_frac_df.columns if cluster_frac_df[ct].sum() > 0]
        colors = [predefined_colors[ct] for ct in used_celltypes]

        # Optional: preview
        print(cluster_frac_df.head())

        import matplotlib.pyplot as plt
        import pandas as pd

        # Assume cluster_frac_df is rows = clusters, columns = cell types
        # cluster_colors: dict mapping cluster index → color

        used_celltypes = [ct for ct in cluster_frac_df.columns if cluster_frac_df[ct].sum() > 0]  # only used
        colors = [predefined_colors[ct] for ct in used_celltypes]

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bar_width = 0.8

        # Plot stacked bars with thin black borders
        cluster_frac_df[used_celltypes].plot(
            kind='bar', stacked=True, ax=ax2, edgecolor='black', linewidth=0.5, color=colors
        )

        # Remove default x-axis labels
        ax2.set_xticks(range(len(cluster_frac_df.index)))
        ax2.set_xticklabels(['' for _ in cluster_frac_df.index])

        # Add colored rectangles below x-axis (tick label position)
        for i, cluster_idx in enumerate(cluster_frac_df.index):
            ax2.annotate(
                '',
                xy=(i, -0.01),  # slightly below x-axis
                xytext=(i + 0.5, -0.01),
                xycoords='data',
                textcoords='data',
                arrowprops=dict(facecolor=cluster_colors.get(cluster_idx, 'grey'), edgecolor='none', width=6, headwidth=0)
            )

        # Adjust bottom margin to show the color strip
        plt.subplots_adjust(bottom=0.15)

        ax2.set_ylabel("Fraction of edges per cell type")
        ax2.set_xlabel("Cluster (module color shown below)")
        ax2.set_title("Cluster composition by cell type")

        # Legend for cell types
        handles = [plt.matplotlib.patches.Patch(color=predefined_colors[ct], label=ct) for ct in used_celltypes]
        ax2.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        #plt.show()
        plt.savefig(os.path.join(os.path.dirname(__file__), "DeepHepatoScope_results", "bar_plot.png"), format="png", dpi=300, bbox_inches="tight")

        # Save to in-memory buffer instead of file
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        buffer.seek(0)
        plt.close()

        # Encode as base64 for HTML rendering
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        image_src = f"data:image/png;base64,{encoded_image}"
        store_bar_image = image_src
        bar_plot_image = html.Img(src=image_src, style={
            "maxWidth": "100%",
            "height": "auto",
            "borderRadius": "10px",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.15)"
        })

        update_bar_plot(bar_plot_image)
        print("--- Cluster composition bar plot saved to DeepHepatoScope_results/bar_plot.png ---")

        #HEATMAP-PLOT

        print(" ")
        print("Calculating gene-gene correlations...")
        update_progress_gcn("80", "Calculating gene-gene correlations...")

        import pandas as pd
        import numpy as np
        from sklearn.decomposition import PCA
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.colors import LinearSegmentedColormap

        # ------------------------------
        # Step 0: Build gene x cell type matrix
        # ------------------------------
        genes = list(G.nodes())
        expr = pd.DataFrame(index=genes, columns=celltypes, dtype=float)
        for g in genes:
            counts = node_edge_ct_counts[g]
            for ct in celltypes:
                expr.loc[g, ct] = counts.get(ct, 0)

        # Remove zero-variance genes
        expr = expr.loc[expr.var(axis=1) > 0]

        # ------------------------------
        # Step 1: Define module colors
        # ------------------------------
        modules = sorted(set(node_cluster.values()))
        module_colors = cluster_colors #{cl: tab20_colors[i % len(tab20_colors)] for cl, i in zip(cluster_labels, selected_indices)}
        row_colors = [module_colors[node_cluster[g]] for g in expr.index]

        fig3, ax3 = plt.subplots(figsize=(15,6))

        # ------------------------------
        # Step 3: Compute module eigengenes (PC1)
        # ------------------------------
        eigengenes = pd.DataFrame(index=modules, columns=celltypes, dtype=float)
        for mod in modules:
            genes_in_mod = [g for g in genes if node_cluster[g] == mod and g in expr.index]
            if len(genes_in_mod) == 1:
                eigengenes.loc[mod] = expr.loc[genes_in_mod].values.flatten()
            else:
                pca = PCA(n_components=1)
                eigengene = pca.fit_transform(expr.loc[genes_in_mod].T)
                eigengenes.loc[mod] = eigengene.flatten()

        # ------------------------------
        # Step 5: Eigengene correlation heatmap with colors
        # ------------------------------
        adj_eig = eigengenes.T.corr()
        sns.clustermap(adj_eig,
                    cmap = LinearSegmentedColormap.from_list("red_white_blue", ["blue", "white", "red"]),
                    center=0,
                    figsize=(6,5),
                    row_colors=[module_colors[m] for m in adj_eig.index],
                    col_colors=[module_colors[m] for m in adj_eig.columns])
        plt.suptitle("Module Eigengene Correlation Heatmap", y=1.02)
        #plt.show()
        plt.savefig(os.path.join(os.path.dirname(__file__), "DeepHepatoScope_results", "correlation_plot.png"), format="png", dpi=300, bbox_inches="tight")

        # Save to in-memory buffer instead of file
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        buffer.seek(0)
        plt.close()

        # Encode as base64 for HTML rendering
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        image_src = f"data:image/png;base64,{encoded_image}"
        store_heatmap_image = image_src
        heatmap_plot_image = html.Img(src=image_src, style={
            "maxWidth": "100%",
            "height": "auto",
            "borderRadius": "10px",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.15)"
        })

        update_heatmap_plot(heatmap_plot_image)
        print("--- Gene-gene correlation plot saved to DeepHepatoScope_results/correlation_plot.png ---")
        
        # node_cluster = {gene: cluster_id, ...}
        cluster_labels = sorted(set(node_cluster.values()))

        # Build and print gene list for each cluster
        # cluster_genes = {}
        # for cl in cluster_labels:
        #     genes_in_cl = [gene for gene, cl_id in node_cluster.items() if cl_id == cl]
        #     cluster_genes[cl] = genes_in_cl
        #     print(f"\nCluster {cl} ({len(genes_in_cl)} genes):")
        #     print("\n".join(genes_in_cl))
    
        # Build gene list for each cluster
        cluster_genes = {}
        data_rows = []
        
        for cl in cluster_labels:
            genes_in_cl = [gene for gene, cl_id in node_cluster.items() if cl_id == cl]
            cluster_genes[cl] = genes_in_cl
            
            # Create rows for table
            for gene in genes_in_cl:
                data_rows.append({
                    'Cluster': cl,
                    'Gene': gene
                })
        
        # Create DataFrame
        df_clusters = pd.DataFrame(data_rows)
        store_module_genes = df_clusters.to_dict('records')
        
        # Create summary for display
        summary_rows = []
        for cl in cluster_labels:
            genes_in_cl = cluster_genes[cl]
            summary_rows.append({
                'Cluster': cl,
                'Gene Count': len(genes_in_cl),
                'Genes': ', '.join(genes_in_cl[:5]) + ('...' if len(genes_in_cl) > 5 else '')
            })
        
        df_summary = pd.DataFrame(summary_rows)
        
        # Build table component
        table_component = html.Div([
            html.H4("Module Genes"),
            html.H4(f"Gene Clusters ({len(cluster_labels)} clusters, {len(data_rows)} genes)"),
            
            # html.Button("Download CSV", id="download-clusters-btn", n_clicks=0),
            # dcc.Download(id="download-clusters-csv"),
            
            html.Br(), html.Br(),
            
            html.H5("Summary View"),
            dash_table.DataTable(
                id='cluster-summary-table',
                columns=[{'name': i, 'id': i} for i in df_summary.columns],
                data=df_summary.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                page_size=10,
                sort_action='native',
            ),
            
            # html.Br(),
            # html.H5("Detailed Gene List"),
            # dash_table.DataTable(
            #     id='cluster-detail-table',
            #     columns=[{'name': i, 'id': i} for i in df_clusters.columns],
            #     data=df_clusters.to_dict('records'),
            #     style_table={'overflowX': 'auto'},
            #     style_cell={
            #         'textAlign': 'left',
            #         'padding': '10px'
            #     },
            #     style_header={
            #         'backgroundColor': 'rgb(230, 230, 230)',
            #         'fontWeight': 'bold'
            #     },
            #     style_data_conditional=[
            #         {
            #             'if': {'row_index': 'odd'},
            #             'backgroundColor': 'rgb(248, 248, 248)'
            #         }
            #     ],
            #     page_size=20,
            #     sort_action='native',
            #     filter_action='native'
            # )
        ])

        all_results = []
        for cl, genes in cluster_genes.items():
            print(f"Running enrichment for cluster {cl} with {len(genes)} genes...")
            try:
                df = enrichr_go(genes)
                df["module"] = cl  # add cluster/module column
                all_results.append(df)
            except Exception as e:
                print(f"⚠️ Cluster {cl} failed: {e}")
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
        else:
            combined_results = pd.DataFrame()


        # ---------------------------
        # Step 1: Load your data
        # ---------------------------
        # file_path = "/content/drive/My Drive/xenium_paths_v2.xlsx"
        # df = pd.read_excel(file_path)

        # Rename 'Module' column to 'module' for consistency
        # df = df.rename(columns={'Module': 'module'})
        combined_results['module'] = pd.to_numeric(combined_results['module'])

        # Compute -log10(Adjusted P-value)
        combined_results['log_adj_p'] = -np.log10(combined_results['Adjusted P-value'])

        # ---------------------------
        # Step 2: Filter and get top 5 per module
        # ---------------------------
        df_filtered = combined_results
        df_filtered = df_filtered.sort_values(by=['module', 'log_adj_p'], ascending=[True, False])
        df_top5 = df_filtered.groupby('module').head(5).reset_index(drop=True)
        store_pathways = df_top5.to_dict('records')

        # ---------------------------
        # Step 3: Map module colors
        # ---------------------------
        module_colors = {mod: cluster_colors[mod] for mod in df_top5['module'].unique() if mod in cluster_colors}
        df_top5['Color'] = df_top5['module'].map(module_colors)

        # ---------------------------
        # Step 4: Plot high-res GSEA-style barplot
        # ---------------------------
        plt.figure(figsize=(12, 10), dpi=1400)  # VERY high DPI

        df_top5['display_label'] = df_top5['Term']
        y_labels = df_top5['display_label']
        x_values = df_top5['log_adj_p']
        y_pos = np.arange(len(df_top5)) * 1.2

        bars = plt.barh(y_pos, x_values, color=df_top5['Color'], edgecolor='black', height=0.8)

        plt.yticks(y_pos, y_labels, fontsize=9)           # small y-axis labels
        plt.xlabel('-log10(Adjusted P-value)', fontsize=12) # smaller x-axis label
        ax.tick_params(axis='x', labelsize=10)  # tick numbers bigger
        plt.gca().invert_yaxis()

        # ---------------------------
        # Step 5: Remove legend & title
        # ---------------------------
        # Do not add legend
        plt.title('')  # remove title

        plt.tight_layout()
        # plt.savefig('gsea_barplot_top5_highres.png', bbox_inches='tight', dpi=1700)
        # plt.show()
        # print("High-resolution GSEA-style barplot saved as gsea_barplot_top5_highres.png")
        plt.savefig(os.path.join(os.path.dirname(__file__), "DeepHepatoScope_results", "enrichr_plot.png"), format="png", dpi=300, bbox_inches="tight")

        # Save to in-memory buffer instead of file
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        buffer.seek(0)
        plt.close()

        # Encode as base64 for HTML rendering
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        image_src = f"data:image/png;base64,{encoded_image}"
        store_enrichr_image = image_src
        enrichr_plot_image = html.Img(src=image_src, style={
            "maxWidth": "100%",
            "height": "auto",
            "borderRadius": "10px",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.15)"
        })

        update_enrichr_plot(enrichr_plot_image)
        print("--- Functional enrichment plot saved to DeepHepatoScope_results/enrichr_plot.png ---")

        normal_button = html.Span([
            html.I(className="bi bi-play-circle-fill me-2"),
            "4. Perform Gene Coexpression Network (GCN) Analysis"
        ])
        
        print(" ")
        print("Returning, check terminal for success message...")
        update_progress_gcn("100", " ")
        return "GCN analysis complete. All plots (GCN, cluster composition, correlation heatmap) have been saved to the DeepHepatoScope_results folder. Check terminal for Enrichr instructions.", {"color": "darkgreen"}, gcn_plot_image, bar_plot_image, heatmap_plot_image, enrichr_plot_image, table_component, store_gcn_image, store_bar_image, store_heatmap_image, store_enrichr_image, store_module_genes, store_pathways, {"display": "block"}, {"display": "block"}, normal_button
    except Exception as e:
        full_traceback = traceback.format_exc()
        return f"Error: {e}\nDetails:\n{full_traceback}", {"color": "red"}, None, None, None

global target_data_sub

@app.callback(
    Output("train-button", "children", allow_duplicate=True),
    Output("train-button", "disabled"),
    Output("run-train-trigger", "data"),
    Input("train-button", "n_clicks"),
    prevent_initial_call=True
)
def start_analysis_ui(n_clicks):
    # Immediately change button to spinner state
    spinner_content = html.Span([
        dbc.Spinner(size="sm", color="light"), #, className="me-2"
        "Analysing..."
    ])
    return spinner_content, True, {"start": True}

@app.callback(
    Output("classification-status", "children"),
    Output("classification-status", "style"),
    Output("spatial-plot-container", "children"),
    Output("pie-plot-container", "children"),
    Output("umap-plot-container", "children"),
    Output("tsne-plot-container", "children"),
    Output("target-data-sub-store", "data", allow_duplicate=True),
    Output("output-df-store", "data", allow_duplicate=True),

    Output("annotations-store", "data", allow_duplicate=True),
    Output("spatial-plot-store", "data", allow_duplicate=True),
    Output("pie-plot-store", "data", allow_duplicate=True),
    Output("umap-plot-store", "data", allow_duplicate=True),
    Output("tsne-plot-store", "data", allow_duplicate=True),

    Output("download-annotations", "style"),
    Output("download-analysis-plots", "style"),

    Output("train-button", "children", allow_duplicate=True),
    Input("run-train-trigger", "data"), #"train-button", "n_clicks"
    State("selected-button", "data"),
    State("model-check", "value"),
    State("model-settings-store", "data"),
    State("target-data-settings-store", "data"),
    State("target-data-store", "data"),
    State("custom-input-store", "data"),
    background=True,
    running=[
        (Output("train-progress-bar", "value"), "0", "100"),
        (Output("train-progress-text", "children"), "Starting analysis...", ""),
        (Output("spatial-plot-container", "children"), None, None),
        (Output("umap-plot-container", "children"), None, None),
        (Output("tsne-plot-container", "children"), None, None)
    ],
    prevent_initial_call=True
)
def train_model(n_clicks, selected_button, model_check, model_settings, target_data_settings, target_data_store, custom_input_store):
    try:
        print("Inference pipeline started. Importing modules...")
        from dash import ctx, set_props
        import os
        import scanpy as sc
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import joblib
        import io
        import time
        print("Loading files and model weights...")
        update_progress("0", "Loading data files...")

        print("Model selected:", model_check)
        # adata_spatial = decode_dict(target_data_store["target_data"])
        adata_spatial = sc.read_h5ad(target_data_store["filename"])
        update_progress("10", "Loading model weights...")
        if model_check=="cosmx":
            #adata_spatial = sc.read_h5ad(r"C:\Users\gnohi\Downloads\ARP2\pub_training\FINAL_codes\CosMx_demo_2775cells_1000genes.h5ad")
            n_genes = 850
            seurat_genes = pd.read_csv(os.path.join(os.path.dirname(__file__), "gene_lists", "final_CosMx850_genes.csv"), header=None).iloc[:, 0].astype(str).tolist()
            model_path = os.path.join(os.path.dirname(__file__), "model_weights", "CosMx_850genes_modelweights_300epochs.pt")
        elif model_check=="xenium":
            n_genes = 360
            seurat_genes = pd.read_csv(os.path.join(os.path.dirname(__file__), "gene_lists", "final_Xenium360_genes.csv"), header=None).iloc[:, 0].astype(str).tolist()
            # seurat_expression_data = sc.read_h5ad(r"C:\Users\gnohi\Downloads\ARP2\pub_training\FINAL_frfr\FINAL_fr_387661_13158_integrated_nocleanlab_float32_asfloat64_Xenium360.h5ad")
            model_path = os.path.join(os.path.dirname(__file__), "model_weights", "Xenium_360genes_modelweights_130epochs.pt")
        elif model_check=="custom":
            n_genes = int(custom_input_store["custom_num_genes"])
            print("number of genes:", n_genes)
            model_path = custom_input_store["custom_model_weights"]
            print("model weights' path:", model_path)
        print("Running log-normalisation...")
        update_progress("20", "Running log-normalisation...")
        set_props("train-progress-bar", {"value": "20"})
        time.sleep(0.5)
        set_props("train-progress-text", {"children": "Running log-normalisation..."})
        time.sleep(0.5)
        sc.pp.normalize_total(adata_spatial, target_sum=1e4)
        sc.pp.log1p(adata_spatial)
        target_data_full = adata_spatial

        seurat_genes_idx = pd.Index(seurat_genes)
        shared_genes = seurat_genes_idx.intersection(adata_spatial.var_names)
        adata_spatial = adata_spatial[:, shared_genes].copy()

        import numpy as np
        X_dense = adata_spatial.X.toarray() if hasattr(adata_spatial.X, "toarray") else adata_spatial.X
        X_dense = normalize(X_dense, axis=1, norm='l2')
        adata_spatial.X = X_dense
        #print("Running...")
        device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
        print("Using device: ", device)

        print("Loading model...")
        update_progress("30", "Loading model...")
        # Set up model and load weights
        input_dim = adata_spatial.shape[1]  # number of genes
        num_classes = ...  # set this to the number of classes in your training data
        model = TransformerClassifier(64, n_genes, 20, ffn_dim=16).to(device)
        #model.load_state_dict(torch.load(r"C:\Users\gnohi\Downloads\ARP2\pub_training\FINAL_fr\final_trained_ssthensl2_CosMx850_100epochs.pt", map_location=torch.device('cpu')))
        if device == "cuda":
            model.load_state_dict(torch.load(model_path))
        elif device == "cpu":
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        #model.load_state_dict(torch.load(r"C:\Users\gnohi\Downloads\trained_ssthenl2_100epochs.pt", map_location=torch.device('cpu')))
        model.eval()
        #print("Running...")
        # Convert to tensor
        X_dense = adata_spatial.X.toarray() if hasattr(adata_spatial.X, "toarray") else adata_spatial.X
        X_dense = np.nan_to_num(X_dense)  # Optional: Replace NaNs with 0
        X_spatial = torch.tensor(X_dense, dtype=torch.float32)

        print("Running model inference...")
        update_progress("40", "Running log-normalisation...")
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm
        X_spatial = X_spatial.to(device)
        # Create DataLoader for batching
        batch_size = 256  # adjust based on GPU memory
        dataset = TensorDataset(X_spatial)
        loader = DataLoader(dataset, batch_size=batch_size)
        # Store predictions
        all_preds = []

        model.eval()
        i = 1
        with torch.no_grad():
            for batch in tqdm(loader):
                #print("Predicting batch:", i)
                x_batch = batch[0].to(device)
                output = model(x_batch)
                # If output is a tuple, take the first item (common for logits)
                if isinstance(output, tuple):
                    output = output[0]
                all_preds.append(output.cpu())
                i += 1
                # if i %1 == 0:
                #     print(i)

        # Combine predictions
        print("Aggregating predictions...")
        update_progress("50", "Aggregating predictions...")
        final_predictions = torch.cat(all_preds, dim=0)
        label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), "label_encoder", "label_encoder_20classes.pkl"))
        predicted_indices = final_predictions.argmax(dim=1).numpy()
        predicted_labels = label_encoder.inverse_transform(predicted_indices)
        predicted_labels_list = predicted_labels.tolist()
        adata_spatial.obs["predicted_cell_type"] = predicted_labels
        print("Predicted labels:", predicted_labels)

        target_data_sub = adata_spatial
        X_target_normalized = adata_spatial.X
        target_data_full.obs['Predicted_Class'] = predicted_labels
        target_data_sub.obs['Predicted_Class'] = predicted_labels
        class_counts = target_data_sub.obs['Predicted_Class'].value_counts()
        print("Predicted label counts:", class_counts)

        output_df = pd.DataFrame({
            "cell_name": target_data_sub.obs_names,
            "predicted_label": target_data_sub.obs["Predicted_Class"]
        })

        new_folder = os.path.join(os.path.dirname(__file__), "DeepHepatoScope_results")
        os.makedirs(new_folder, exist_ok=True)

        output_df.to_csv(os.path.join(os.path.dirname(__file__), "DeepHepatoScope_results", "DeepHepatoScope_annotations.csv"))
        store_annotations = output_df.to_dict("records")

        print(" ")
        print("--- INFERENCE COMPLETE! DEEPHEPATOSCOPE ANNOTATIONS SAVED SUCCESSFULLY TO: ---")
        print("DeepHepatoScope_results/DeepHepatoScope_annotations.csv")
        print(" ")
        
        update_progress("60", "Inference complete. Moving onto spatial plotting...")

        output_df_str = output_df.to_json(orient="records")

        predefined_colors = {
            # Existing entries (preserved)
            "Malignant cells": "#aec7e8",
            "Stellate cells": "#1f77b4",
            "TAMs": "#ff7f0e",
            "T cells": "#ffbb78",
            "TECs": "#2ca02c",
            "B cells": "#98df8a",
            "Erythroid cells": "#d62728",
            "Hepatocytes": "#ff9896",
            "NK cells": "#9467bd",
            "Cholangiocytes": "#c5b0d5",
            "Unclassified": "#8c564b",
            # Newly added entries
            "Mast cells": "#e377c2",                     # magenta
            "Dendritic cells": "#7f7f7f",                 # grey
            "CAFs": "#bcbd22",                            # olive
            "Myofibroblasts": "#17becf",                  # cyan
            "Monocytes": "#c49c94",                       # light brown
            "GMPs": "#f7b6d2",                            # pale pink
            "Pericytes": "#8c6d31",                       # brownish
            "Smooth muscle cells": "#393b79",             # dark blue
            "Neutrophils": "#9edae5",                     # light cyan
            "Plasma cells": "#c7c7c7",                    # silver
            "Plasmacytoid dendritic cells": "#636363",    # dark grey
            "Others": "#8c8c8c"
        }
        target_data_full.obs["Predicted_Class"] = target_data_full.obs["Predicted_Class"].astype("category")
        # Set the colors in the AnnData object's uns attribute
        target_data_full.uns["Predicted_Class_colors"] = [
            predefined_colors.get(cat, "#000000")  # Default to black if color not found
            for cat in target_data_full.obs["Predicted_Class"].cat.categories
]

        import io
        import base64
        import matplotlib.pyplot as plt
        import scanpy as sc

        # Map each predicted class to a unique color
        unique_classes = target_data_sub.obs['Predicted_Class'].unique()
        print("Unique classes of labels:", unique_classes)

        import matplotlib.pyplot as plt
        import io
        import base64

        print(" ")
        print("Attempting to plot spatial plot using coordinates in .h5ad file...")
        if "X_spatial" in target_data_sub.obs.columns and "Y_spatial" in target_data_sub.obs.columns:

            target_coordinates = pd.concat(
                [target_data_sub.obs["X_spatial"], target_data_sub.obs["Y_spatial"]], axis=1
            )
            target_coordinates['scaled_x'] = target_coordinates['X_spatial']
            target_coordinates['scaled_y'] = target_coordinates['Y_spatial']
            target_coordinates['cell'] = target_data_sub.obs.index
            
            # Map Predicted_Class and class colors
            mapping_dict = dict(zip(target_data_sub.obs.index, target_data_sub.obs['Predicted_Class']))
            target_coordinates['class'] = target_coordinates['cell'].map(mapping_dict)
            target_coordinates['color'] = target_coordinates['class'].map(predefined_colors)

            # Matplotlib scatter plot
            plt.figure(figsize=(6, 6), dpi=100)
            for cls, group in target_coordinates.groupby('class'):
                plt.scatter(
                    group['scaled_x'],
                    group['scaled_y'],
                    s=0.3,  # point size
                    c=group['color'], #class_colors_hex.get(cls, "#000000"),  # fallback black
                    label=cls,
                    alpha=0.8
                )

            plt.title("Spatial Plot")
            plt.legend(markerscale=2, fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.axis("equal")
            plt.axis("off")
            plt.tight_layout()

            plt.savefig(os.path.join(os.path.dirname(__file__), "DeepHepatoScope_results", "spatial_plot.png"), format="png", dpi=300, bbox_inches="tight")

            
            # Save to in-memory buffer instead of file
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            buffer.seek(0)
            plt.close()

            # Encode as base64 for HTML rendering
            encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
            image_src = f"data:image/png;base64,{encoded_image}"
            store_spatial_image = image_src
            spatial_image = html.Img(src=image_src, style={
                "maxWidth": "100%",
                "height": "auto",
                "borderRadius": "10px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.15)"
            })

            update_spatial_plot(spatial_image)
            print("--- Spatial plot saved to DeepHepatoScope_results/spatial_plot.png ---")
            update_progress("70", "Spatial plotting complete. Moving onto dimensionality reduction plotting...")
        
        else:
            spatial_image = None
            print("--- WARNING: No spatial coordinates found! ---")
            print("--- WARNING: You are either working with scRNA-Seq data (in which case this is intended), our your spatial coordinates columns are not labelled X_spatial and Y_spatial ---")
            print("--- WARNING: Skipping spatial plot! ---")
        
        # Count occurrences of each predicted label
        label_counts = target_data_full.obs["Predicted_Class"].value_counts()

        # Match colors to labels (default gray for missing ones)
        colors = [predefined_colors.get(label, "#cccccc") for label in label_counts.index]

        # Plot pie chart
        # Create pie chart
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        # Temporarily compute percentages but don't display them on the pie
        wedges, texts, autotexts = ax.pie(
            label_counts.values,   # no parentheses if it's already a Series
            labels=None,           # no labels on the pie itself
            colors=colors[:len(label_counts)],
            autopct='%1.1f%%',     # used internally to get pct text
            startangle=90,
            counterclock=False
        )

        # Remove percentage texts from the pie
        for t in autotexts:
            t.set_visible(False)
        # Add legend on the right with class names and percentages
        labels = [
            f"{cls} ({pct.get_text()})"
            for cls, pct in zip(label_counts.keys(), autotexts)
        ]
        ax.legend(
            wedges,
            labels,
            title="Classes",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=10
        )
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "DeepHepatoScope_results", "celltype_composition_pie_plot.png"), format="png", dpi=300, bbox_inches="tight")
        # Save to in-memory buffer instead of file
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        buffer.seek(0)
        plt.close()

        # Encode as base64 for HTML rendering
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        image_src = f"data:image/png;base64,{encoded_image}"
        store_pie_image = image_src
        pie_image = html.Img(src=image_src, style={
            "maxWidth": "100%",
            "height": "auto",
            "borderRadius": "10px",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.15)"
        })
        update_pie_plot(pie_image)
        print("--- Cell type compostion pie chart saved to DeepHepatoScope_results/celltype_composition_pie_plot.png ---")

        sc.set_figure_params(dpi=600)
        sc.set_figure_params(figsize=(10, 8))

        if not target_data_settings.get("calculation"):
            umap_image = None
            store_umap_image = None
        elif 'umap' not in target_data_settings["calculation"]:
            umap_image = None
            store_umap_image = None
        elif 'umap' in target_data_settings["calculation"] and "X_UMAP" not in target_data_full.obsm:
            print(" ")
            print("Calculating UMAP...")
            sc.pp.highly_variable_genes(target_data_full, n_top_genes = target_data_settings["variable_genes"])
            sc.tl.pca(target_data_full, n_comps=target_data_settings["pca_dims"])
            sc.pp.neighbors(target_data_full)
            sc.tl.umap(target_data_full)
            title = "UMAP Visualization of Predicted Classes"
            sc.pl.umap(
                target_data_full,
                color="Predicted_Class",
                title=title,
                legend_loc="right margin",   # or "right margin"
                frameon=False,
                size=10,
                show=False              # <- don't display interactively
            )
            plt.savefig(os.path.join(os.path.dirname(__file__), "DeepHepatoScope_results", "umap_plot.png"), format="png", dpi=300, bbox_inches="tight")

            # Save to in-memory buffer instead of file
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            buffer.seek(0)
            plt.close()

            # Encode as base64 for HTML rendering
            encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
            image_src = f"data:image/png;base64,{encoded_image}"
            store_umap_image = image_src
            umap_image = html.Img(src=image_src, style={
                "maxWidth": "100%",
                "height": "auto",
                "borderRadius": "10px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.15)"
            })

            update_umap_plot(umap_image)
            print("--- UMAP plot saved to DeepHepatoScope_results/umap_plot.png ---")
            update_progress("80", "UMAP plotting complete. Moving onto tSNE plotting (if requested)...")

        if not target_data_settings.get("calculation"):
            tsne_image = None
            store_tsne_image = None
        elif 'tsne' not in target_data_settings["calculation"]:
            tsne_image = None
            store_tsne_image = None
        elif 'tsne' in target_data_settings["calculation"] and "X_TSNE" not in target_data_full.obsm:
            print(" ")
            print("Calculating TSNE...")
            sc.pp.highly_variable_genes(target_data_full, n_top_genes = target_data_settings["variable_genes"])
            sc.tl.pca(target_data_full, n_comps=target_data_settings["pca_dims"])
            sc.pp.neighbors(target_data_full)
            sc.tl.tsne(target_data_full)
            title = "t-SNE Visualization of Predicted Classes"
            sc.pl.tsne(
                target_data_full,
                color="Predicted_Class",
                title=title,
                legend_loc="right margin",
                frameon=False,
                size=10,
                show=False
            )
            plt.savefig(os.path.join(os.path.dirname(__file__), "DeepHepatoScope_results", "tsne_plot.png"), format="png", dpi=300, bbox_inches="tight")

            # Save to in-memory buffer instead of file
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            buffer.seek(0)
            plt.close()

            # Encode as base64 for HTML rendering
            encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
            image_src = f"data:image/png;base64,{encoded_image}"
            store_tsne_image = image_src
            tsne_image = html.Img(src=image_src, style={
                "maxWidth": "100%",
                "height": "auto",
                "borderRadius": "10px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.15)"
            })

            update_tsne_plot(tsne_image)
            print("--- TSNE plot saved to DeepHepatoScope_results/tsne_plot.png ---")
            update_progress("90", "tSNE plotting complete. Wrapping up...")
        
        normal_button = html.Span([
            html.I(className="bi bi-play-circle-fill me-2"),
            "3. Analyse and Annotate Target Cell Types"
        ])

        print(" ")
        print("Returning, check terminal for success message...")
        update_progress("100", "Analysis complete!")
        return "Model trained successfully. CSV of annotations and all plots (spatial, UMAP/tSNE) have been saved to the DeepHepatoScope_results folder.", {"color": "darkgreen"}, spatial_image, pie_image, umap_image, tsne_image, None, {'output_df_str': output_df_str, 'predicted_labels_list': predicted_labels_list}, store_annotations, store_spatial_image, store_pie_image, store_umap_image, store_tsne_image, {"display": "block"}, {"display": "block"}, normal_button #{'target_data_sub': target_data_sub_dict}, {'output_df_str': output_df_str}

    except Exception as e:
        full_traceback = traceback.format_exc()
        return f"Error: {e}\nDetails:\n{full_traceback}", {"color": "red"}, None, None, None, {'target_data_sub': None}, {'output_df_str': None}

# Run the app
if __name__ == "__main__":
    app.run(debug=True) #Replaced run_server
