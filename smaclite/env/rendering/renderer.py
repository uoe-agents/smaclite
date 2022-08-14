from typing import List

import pygame
from smaclite.env.maps.map import MapInfo
from smaclite.env.terrain.terrain import TerrainType
from smaclite.env.units.combat_type import CombatType
from smaclite.env.units.unit import Unit
from smaclite.env.units.unit_command import MoveCommand
from smaclite.env.units.unit_type import StandardUnit
from smaclite.env.util.faction import Faction


TILE_SIZE = 32
TERRAIN_COLORS = {
    TerrainType.NORMAL: (214, 229, 226),
    TerrainType.CLIFF: (255, 0, 0),
    TerrainType.NONE: (182, 213, 216),
}
FACTION_COLORS = {
    Faction.ALLY: (56, 179, 73),
    Faction.ENEMY: (232, 93, 96),
}
UNIT_TYPE_ABBREVIATIONS = {
    StandardUnit.BANELING: "bnl",
    StandardUnit.COLOSSUS: "cls",
    StandardUnit.MARAUDER: "mrd",
    StandardUnit.MARINE: "mrn",
    StandardUnit.MEDIVAC: "mdv",
    StandardUnit.STALKER: "stl",
    StandardUnit.ZEALOT: "zlt",
    StandardUnit.ZERGLING: "zrg",
}
# Equivalent to the "faster" in-game speed
RENDER_FPS = 22.4


class Renderer:
    def __init__(self):
        self.window = None
        self.clock = None
        self.fonts = {}

    def render(self, map_info: MapInfo, units: List[Unit]):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("SMAClite")
            self.window = self.__create_window(map_info)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((TILE_SIZE * map_info.width,
                                 TILE_SIZE * map_info.height),
                                pygame.SRCALPHA)
        for x in range(map_info.width):
            for y in (range(map_info.height)):
                real_y = map_info.height - y - 1
                terrain_type = map_info.terrain[y][x]
                color = TERRAIN_COLORS[terrain_type]

                pygame.draw.rect(canvas, color,
                                 pygame.Rect((TILE_SIZE * x,
                                              TILE_SIZE * real_y),
                                             (TILE_SIZE, TILE_SIZE))),

        for unit in sorted(units, key=lambda u: u.plane.z):
            if unit.hp == 0:
                continue
            color = FACTION_COLORS[unit.faction]
            radius = TILE_SIZE * unit.radius
            center = np_to_pygame(unit.pos, map_info.height)
            pygame.draw.circle(canvas, color, center, radius,
                               width=2)
            pygame.draw.circle(canvas, tuple(0.6 * c for c in color),
                               center, radius - 2)
            circle_height = unit.hp / unit.max_hp * 2 * radius
            clip_area = pygame.Rect(center[0] - radius,
                                    center[1] + radius - circle_height,
                                    2 * radius, circle_height)
            canvas.set_clip(clip_area)
            pygame.draw.circle(canvas, color, center, radius)
            canvas.set_clip(None)
            main_font_size = int(radius * 0.9)
            if unit.type not in self.fonts:
                text = UNIT_TYPE_ABBREVIATIONS.get(unit.type, "CST")
                font = pygame.font.SysFont("Monospace", main_font_size) \
                    .render(text, True, (0, 0, 0))
                self.fonts[unit.type] = font
            font = self.fonts[unit.type]
            hp_str = f"{unit.hp}"
            if unit.max_shield:
                hp_str += f"+{unit.shield}"
            hp_font = pygame.font.SysFont("Monospace",
                                          int(radius) // 2) \
                .render(hp_str, True, (0, 0, 0), (255, 255, 255, 230))
            cd_text = unit.cooldown if unit.combat_type == CombatType.DAMAGE \
                else unit.energy
            cd_font = pygame.font.SysFont("Monospace",
                                          int(radius) // 2) \
                .render(f"{cd_text:.02f}", True, (0, 0, 0),
                        (255, 255, 255, 230))
            canvas.blit(font, font.get_rect(center=center))
            canvas.blit(hp_font,
                        hp_font.get_rect(center=(center[0],
                                                 center[1] + main_font_size)))
            canvas.blit(cd_font,
                        cd_font.get_rect(center=(center[0],
                                                 center[1] - main_font_size)))
            if unit.target is not None:
                color = FACTION_COLORS[unit.faction]
                pygame.draw.line(canvas, color,
                                 np_to_pygame(unit.pos, map_info.height),
                                 np_to_pygame(unit.target.pos,
                                              map_info.height))
            elif isinstance(unit.command, MoveCommand):
                pygame.draw.line(canvas, (0, 0, 255),
                                 np_to_pygame(unit.pos, map_info.height),
                                 np_to_pygame(unit.command.pos,
                                              map_info.height))

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(RENDER_FPS)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def __create_window(self, map_info: MapInfo):
        return pygame.display.set_mode((TILE_SIZE * map_info.width,
                                        TILE_SIZE * map_info.height))


def np_to_pygame(np_vec, height):
    return (TILE_SIZE * np_vec[0].astype(float),
            TILE_SIZE * (height - np_vec[1].astype(float)))
