from matplotlib.colors import ListedColormap, to_rgb


def get_cmap(name):
    assert name in globals(), f"Colormap with name {name} unknown."
    hex_colors = globals()[name]

    # hex to rgb
    rgb_colos = [to_rgb(h) for h in hex_colors]

    return ListedColormap(rgb_colos)


# orig reference http://epub.wu.ac.at/1692/1/document.pdf
zeileis26 = [
    "#023fa5",
    "#7d87b9",
    "#bec1d4",
    "#d6bcc0",
    "#bb7784",
    "#8e063b",
    "#4a6fe3",
    "#8595e1",
    "#b5bbe3",
    "#e6afb9",
    "#e07b91",
    "#d33f6a",
    "#11c638",
    "#8dd593",
    "#c6dec7",
    "#ead3c6",
    "#f0b98d",
    "#ef9708",
    "#0fcfc0",
    "#9cded6",
    "#d5eae7",
    "#f3e1eb",
    "#f6c4e1",
    "#f79cd4",
    # these last ones were added:
    "#7f7f7f",
    "#c7c7c7",
]

wandb16 = [
    "#5387DD",
    "#DA4C4C",
    "#479A5F",
    "#7D54B2",
    "#E87B9F",
    "#E57439",
    "#87CEBF",
    "#C565C7",
    "#EDB732",
    "#5BC5DB",
    "#A0C75C",
    "#A12864",
    "#F0B899",
    "#229487",
    "#A46750",
    "#A1A9AD",
]


# theory - https://eleanormaclure.files.wordpress.com/2011/03/colour-coding.pdf (page 5)
# kelly's colors - https://i.kinja-img.com/gawker-media/image/upload/1015680494325093012.JPG
# hex values - http://hackerspace.kinja.com/iscc-nbs-number-hex-r-g-b-263-f2f3f4-242-243-244-267-22-1665795040
kelly20 = [
    "#F3C300",
    "#875692",
    "#F38400",
    "#A1CAF1",
    "#BE0032",
    "#C2B280",
    "#848482",
    "#008856",
    "#E68FAC",
    "#0067A5",
    "#F99379",
    "#604E97",
    "#F6A600",
    "#B3446C",
    "#DCD300",
    "#882D17",
    "#8DB600",
    "#654522",
    "#E25822",
    "#2B3D26",
]

# gothenburg palette
# https://eleanormaclure.files.wordpress.com/2011/03/colour-coding.pdf
gothenburg = [
    "#DC8F8F",
    "#4A8992",
    "#A5BA39",
    "#D33227",
    "#EFE945",
    "#46479D",
    "#D96733",
    "#8B3D93",
    "#3660AC",
    "#7FB484",
    "#AF8FC2",
    "#E2A932",
    "#90C7EB",
    "#49823D",
    "#675722",
    "#672C26",
]
