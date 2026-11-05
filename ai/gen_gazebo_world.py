import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import CLASSES, CUTOUT_DIR, PROJECT_ROOT, TEXTURE_DIR

PACKAGE_DIR = PROJECT_ROOT / 'ros2_ws' / 'src' / 'uav_waste_sim'
MODELS_DIR = PACKAGE_DIR / 'models'
WORLD_PATH = PACKAGE_DIR / 'worlds' / 'waste_field.sdf'
LAYOUT_PATH = PACKAGE_DIR / 'config' / 'waste_layout.json'

REAL_LENGTH_M = {
    'bottle': (0.24, 0.32),
    'plastic_bag': (0.32, 0.45),
    'battery': (0.10, 0.14),
    'organic_waste': (0.20, 0.30),
}

GROUND_SIZE = 40.0
GROUND_TILE_M = 2.5


def write_ground_mesh(path, size, tile):
    half = size / 2.0
    repeat = size / tile
    path.write_text(
        'o ground\n'
        f'v {-half} {-half} 0\nv {half} {-half} 0\nv {half} {half} 0\nv {-half} {half} 0\n'
        f'vt 0 0\nvt {repeat} 0\nvt {repeat} {repeat}\nvt 0 {repeat}\n'
        'vn 0 0 1\n'
        'f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n',
        encoding='utf-8')


def build_ground_model(texture_name):
    model_dir = MODELS_DIR / 'waste_field_ground'
    if model_dir.exists():
        shutil.rmtree(model_dir)
    (model_dir / 'meshes').mkdir(parents=True)
    (model_dir / 'materials' / 'textures').mkdir(parents=True)
    write_ground_mesh(model_dir / 'meshes' / 'ground.obj', GROUND_SIZE, GROUND_TILE_M)
    texture = Image.open(TEXTURE_DIR / f'{texture_name}.jpg').convert('RGB').resize((1024, 1024))
    texture.save(model_dir / 'materials' / 'textures' / 'ground.jpg', quality=92)
    (model_dir / 'model.config').write_text(
        '<?xml version="1.0"?>\n<model>\n  <name>waste_field_ground</name>\n  <version>1.0</version>\n'
        '  <sdf version="1.9">model.sdf</sdf>\n  <description>Mặt đất bãi rác, texture CC0 ambientCG</description>\n</model>\n',
        encoding='utf-8')
    (model_dir / 'model.sdf').write_text(f'''<?xml version="1.0"?>
<sdf version="1.9">
  <model name="waste_field_ground">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <plane><normal>0 0 1</normal><size>{GROUND_SIZE} {GROUND_SIZE}</size></plane>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh><uri>model://waste_field_ground/meshes/ground.obj</uri></mesh>
        </geometry>
        <material>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
          <pbr>
            <metal>
              <albedo_map>model://waste_field_ground/materials/textures/ground.jpg</albedo_map>
              <roughness>0.95</roughness>
              <metalness>0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>
''', encoding='utf-8')


def keep_largest_component(image):
    pixels = np.array(image)
    mask = (pixels[:, :, 3] > 128).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        return image
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pixels[labels != largest, 3] = 0
    ys, xs = np.where(pixels[:, :, 3] > 128)
    return Image.fromarray(pixels, 'RGBA').crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def bleed_transparent_pixels(image):
    pixels = np.array(image)
    opaque = pixels[:, :, 3] > 128
    if opaque.any():
        mean_color = pixels[opaque][:, :3].mean(axis=0).astype(np.uint8)
        pixels[~opaque, :3] = mean_color
    return Image.fromarray(pixels, 'RGBA')


def place_objects(rng, count, area_half, min_gap):
    sim_cutouts = {name: sorted((CUTOUT_DIR / 'sim' / name).glob('*.png')) for name in CLASSES}
    for name, files in sim_cutouts.items():
        if not files:
            raise SystemExit(f'Thiếu cutout sim/{name}')
    texture_dir = MODELS_DIR / 'waste_items' / 'materials' / 'textures'
    if texture_dir.parent.parent.exists():
        shutil.rmtree(texture_dir.parent.parent)
    texture_dir.mkdir(parents=True)
    (MODELS_DIR / 'waste_items' / 'model.config').write_text(
        '<?xml version="1.0"?>\n<model>\n  <name>waste_items</name>\n  <version>1.0</version>\n'
        '  <description>Texture vật thể rác dùng trong world waste_field</description>\n</model>\n',
        encoding='utf-8')

    objects = []
    attempts = 0
    while len(objects) < count and attempts < count * 50:
        attempts += 1
        class_name = CLASSES[len(objects) % len(CLASSES)]
        x = rng.uniform(-area_half, area_half)
        y = rng.uniform(-area_half, area_half)
        if any(math.hypot(x - o['x'], y - o['y']) < min_gap for o in objects):
            continue
        source = rng.choice(sim_cutouts[class_name])
        image = keep_largest_component(Image.open(source).convert('RGBA'))
        length = rng.uniform(*REAL_LENGTH_M[class_name])
        if image.width >= image.height:
            size_x, size_y = length, length * image.height / image.width
        else:
            size_x, size_y = length * image.width / image.height, length
        name = f'{class_name}_{len(objects):02d}'
        bleed_transparent_pixels(image).save(texture_dir / f'{name}.png')
        objects.append({
            'name': name,
            'class_name': class_name,
            'x': round(x, 3),
            'y': round(y, 3),
            'yaw': round(rng.uniform(-math.pi, math.pi), 3),
            'size_x': round(size_x, 3),
            'size_y': round(size_y, 3),
            'texture': f'model://waste_items/materials/textures/{name}.png',
            'source_cutout': source.name,
        })
    return objects


def object_model_sdf(obj):
    return f'''
    <model name="{obj['name']}">
      <static>true</static>
      <pose>{obj['x']} {obj['y']} 0.004 0 0 {obj['yaw']}</pose>
      <link name="link">
        <visual name="visual">
          <cast_shadows>false</cast_shadows>
          <geometry>
            <plane><normal>0 0 1</normal><size>{obj['size_x']} {obj['size_y']}</size></plane>
          </geometry>
          <material>
            <diffuse>1 1 1 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
            <pbr>
              <metal>
                <albedo_map>{obj['texture']}</albedo_map>
                <roughness>0.7</roughness>
                <metalness>0</metalness>
              </metal>
            </pbr>
          </material>
        </visual>
      </link>
    </model>'''


def world_sdf(objects, uav_x, uav_y):
    models = ''.join(object_model_sdf(obj) for obj in objects)
    return f'''<?xml version="1.0"?>
<sdf version="1.9">
  <world name="waste_field">
    <physics name="4ms" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <scene>
      <ambient>0.75 0.75 0.75 1</ambient>
      <background>0.62 0.78 0.92 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.95 0.93 0.88 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.35 0.2 -0.9</direction>
    </light>

    <include>
      <uri>model://waste_field_ground</uri>
    </include>

    <include>
      <name>x3_uav</name>
      <uri>model://x3_uav</uri>
      <pose>{uav_x} {uav_y} 0.06 0 0 0</pose>
    </include>
{models}
  </world>
</sdf>
'''


def main():
    parser = argparse.ArgumentParser(description='Sinh world Gazebo bãi rác với vật thể từ cutout tập sim')
    parser.add_argument('--objects', type=int, default=64)
    parser.add_argument('--area', type=float, default=5.5)
    parser.add_argument('--min-gap', type=float, default=0.5)
    parser.add_argument('--ground', default='Grass004')
    parser.add_argument('--seed', type=int, default=21)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    build_ground_model(args.ground)
    objects = place_objects(rng, args.objects, args.area, args.min_gap)
    WORLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORLD_PATH.write_text(world_sdf(objects, -args.area - 1.5, 0.0), encoding='utf-8')
    LAYOUT_PATH.write_text(json.dumps({'ground': args.ground, 'area_half': args.area, 'objects': objects},
                                      ensure_ascii=False, indent=2), encoding='utf-8')
    counts = {name: sum(1 for o in objects if o['class_name'] == name) for name in CLASSES}
    print('world ->', WORLD_PATH, counts)


if __name__ == '__main__':
    main()
