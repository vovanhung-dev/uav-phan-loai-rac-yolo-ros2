WASTE_CLASSES = ['bottle', 'plastic_bag', 'battery', 'organic_waste']

WASTE_LABELS_VI = {
    'bottle': 'Chai',
    'plastic_bag': 'Tui nilon',
    'battery': 'Pin',
    'organic_waste': 'Rac huu co',
}

WASTE_COLORS_BGR = {
    'bottle': (255, 170, 0),
    'plastic_bag': (255, 0, 200),
    'battery': (0, 40, 255),
    'organic_waste': (0, 200, 60),
}


def color_for(class_name):
    return WASTE_COLORS_BGR.get(class_name, (200, 200, 200))
