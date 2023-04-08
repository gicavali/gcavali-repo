import geopandas as gp
import networkx as nx
import pandas as pd

G = nx.Graph()
geo_city_df = gp.read_file("BR_Municipios_2020/BR_Municipios_2020.shp")
geo_city_df["MUN_UF"] = geo_city_df["NM_MUN"] + " (" + geo_city_df["SIGLA_UF"] + ")"
geo_city_df.head(2)

# Creating 2 undirected networke where each node is a city that
# is linked to its neighboors.

# G = network roads
# H = network roads + airways

for index, row in geo_city_df.iterrows():
    neighbors = geo_city_df[
        geo_city_df.geometry.touches(row["geometry"])
    ].MUN_UF.tolist()
    neighbors = [
        (row.MUN_UF, name, {"aereo": "no"}) for name in neighbors if row.MUN_UF != name
    ]
    G.add_edges_from(neighbors)
    nx.set_node_attributes(
        G,
        {row.MUN_UF: {"long": row.geometry.centroid.x, "lat": row.geometry.centroid.y}},
    )

flights_df = pd.read_excel("airport_bz.xlsx")

H = G.copy()
for index, row in flights_df.iterrows():
    H.add_edge(row.A1, row.A2, aereo="yes")

# Adding node attributes:
# population + population class, region, capita

pop_df = pd.read_excel("MUNIC_2019.xlsx")

for index, row in pop_df.iterrows():
    nx.set_node_attributes(
        G,
        {row["NOME MUNIC"]: {"population": row["POP EST"], "class": row["CLASSE POP"]}},
    )
    nx.set_node_attributes(
        H,
        {row["NOME MUNIC"]: {"population": row["POP EST"], "class": row["CLASSE POP"]}},
    )

node_dict = {
    "Tabocão (TO)": 2423,
    "Ereré (CE)": 7225,
    "Olho d'Água do Borges (RN)": 4244,
    "Campo Grande (RN)": 9768,
    "São Vicente Férrer (PE)": 18085,
    "Amparo do São Francisco (SE)": 2380,
    "Gracho Cardoso (SE)": 5824,
    "Araçás (BA)": 12208,
    "Santa Terezinha (BA)": 10464,
    "Muquém do São Francisco (BA)": 11417,
    "Iuiu (BA)": 11016,
    "Dona Euzébia (MG)": 6619,
    "São Tomé das Letras (MG)": 7120,
    "Passa Vinte (MG)": 2031,
    "Atílio Vivácqua (ES)": 12105,
    "Itaoca (SP)": 3330,
    "Biritiba Mirim (SP)": 32936,
    "Florínea (SP)": 2653,
    "São Luiz do Paraitinga (SP)": 10690,
    "Lauro Müller (SC)": 15313,
    "Grão-Pará (SC)": 6595,
    "São Cristóvão do Sul (SC)": 5598,
}
nx.set_node_attributes(G, node_dict, "population")
nx.set_node_attributes(H, node_dict, "population")
node_dict = {
    "Tabocão (TO)": "2 - 5001 até 10000",
    "Ereré (CE)": "2 - 5001 até 10000",
    "Olho d'Água do Borges (RN)": "2 - 5001 até 10000",
    "Campo Grande (RN)": "2 - 5001 até 10000",
    "São Vicente Férrer (PE)": "3 - 10001 até 20000",
    "Amparo do São Francisco (SE)": "2 - 5001 até 10000",
    "Gracho Cardoso (SE)": "2 - 5001 até 10000",
    "Araçás (BA)": "3 - 10001 até 20000",
    "Santa Terezinha (BA)": "3 - 10001 até 20000",
    "Muquém do São Francisco (BA)": "3 - 10001 até 20000",
    "Iuiu (BA)": "3 - 10001 até 20000",
    "Dona Euzébia (MG)": "2 - 5001 até 10000",
    "São Tomé das Letras (MG)": "2 - 5001 até 10000",
    "Passa Vinte (MG)": "2 - 5001 até 10000",
    "Atílio Vivácqua (ES)": "3 - 10001 até 20000",
    "Itaoca (SP)": "2 - 5001 até 10000",
    "Biritiba Mirim (SP)": "4 - 20001 até 50000",
    "Florínea (SP)": "2 - 5001 até 10000",
    "São Luiz do Paraitinga (SP)": "3 - 10001 até 20000",
    "Lauro Müller (SC)": "3 - 10001 até 20000",
    "Grão-Pará (SC)": "2 - 5001 até 10000",
    "São Cristóvão do Sul (SC)": "2 - 5001 até 10000",
}
nx.set_node_attributes(G, node_dict, "class")
nx.set_node_attributes(H, node_dict, "class")
nx.set_node_attributes(H, {"Fernando de Noronha (PE)": -3.853808}, "lat")
nx.set_node_attributes(H, {"Fernando de Noronha (PE)": -32.423786}, "long")

for node in H.nodes():
    if node[-3:][:2] in ["PR", "SC", "RS"]:
        nx.set_node_attributes(H, {node: "Sul"}, "region")
    elif node[-3:][:2] in ["SP", "MG", "RJ", "ES"]:
        nx.set_node_attributes(H, {node: "Sudeste"}, "region")
    elif node[-3:][:2] in ["MT", "MS", "GO", "DF"]:
        nx.set_node_attributes(H, {node: "CentroOeste"}, "region")
    elif node[-3:][:2] in ["BA", "PI", "MA", "SE", "AL", "PE", "PB", "RN", "CE"]:
        nx.set_node_attributes(H, {node: "Nordeste"}, "region")
    elif node[-3:][:2] in ["AC", "RO", "AM", "RR", "PA", "AP", "TO"]:
        nx.set_node_attributes(H, {node: "Norte"}, "region")

    if node in [
        "Manaus (AM)",
        "Boa Vista (RR)",
        "Macapá (AP)",
        "Belém (PA)",
        "Palmas (TO)",
        "Porto Velho (RO)",
        "Rio Branco (AC)",
        "São Luís (MA)",
        "Fortaleza (CE)",
        "Natal (RN)",
        "Recife (PE)",
        "João Pessoa (PB)",
        "Aracaju (SE)",
        "Maceió (AL)",
        "Salvador (BA)",
        "Cuiabá (MT)",
        "Campo Grande (MS)",
        "Goiânia (GO)",
        "São Paulo (SP)",
        "Rio de Janeiro (RJ)",
        "Vitória (ES)",
        "Belo Horizonte (MG)",
        "Curitiba (PR)",
        "Porto Alegre (RS)",
        "Florianópolis (SC)",
    ]:
        nx.set_node_attributes(H, {node: "yes"}, "capital")
    else:
        nx.set_node_attributes(H, {node: "no"}, "capital")

for node in G.nodes():
    if node[-3:][:2] in ["PR", "SC", "RS"]:
        nx.set_node_attributes(G, {node: "Sul"}, "region")
    elif node[-3:][:2] in ["SP", "MG", "RJ", "ES"]:
        nx.set_node_attributes(G, {node: "Sudeste"}, "region")
    elif node[-3:][:2] in ["MT", "MS", "GO"]:
        nx.set_node_attributes(G, {node: "CentroOeste"}, "region")
    elif node[-3:][:2] in ["BA", "PI", "MA", "SE", "AL", "PE", "PB", "RN", "CE"]:
        nx.set_node_attributes(G, {node: "Nordeste"}, "region")
    elif node[-3:][:2] in ["AC", "RO", "AM", "RR", "PA", "AP", "TO"]:
        nx.set_node_attributes(G, {node: "Norte"}, "region")

    if node in [
        "Manaus (AM)",
        "Boa Vista (RR)",
        "Macapá (AP)",
        "Belém (PA)",
        "Palmas (TO)",
        "Porto Velho (RO)",
        "Rio Branco (AC)",
        "São Luís (MA)",
        "Fortaleza (CE)",
        "Natal (RN)",
        "Recife (PE)",
        "João Pessoa (PB)",
        "Aracaju (SE)",
        "Maceió (AL)",
        "Salvador (BA)",
        "Cuiabá (MT)",
        "Campo Grande (MS)",
        "Goiânia (GO)",
        "São Paulo (SP)",
        "Rio de Janeiro (RJ)",
        "Vitória (ES)",
        "Belo Horizonte (MG)",
        "Curitiba (PR)",
        "Porto Alegre (RS)",
        "Florianópolis (SC)",
    ]:
        nx.set_node_attributes(G, {node: "yes"}, "capital")
    else:
        nx.set_node_attributes(G, {node: "no"}, "capital")

# Saving networks
nx.write_gexf(H, "grafo_cidades_aeroportos.gexf")
nx.write_gexf(G, "grafo_cidades.gexf")
