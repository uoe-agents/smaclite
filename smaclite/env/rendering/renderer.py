

from typing import List
from smaclite.env.maps.map import Faction, MapInfo, TerrainType
import pygame

from smaclite.env.units.unit import Unit
from smaclite.env.units.unit_command import AttackMoveCommand, MoveCommand
from smaclite.env.units.unit_type import UnitType

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
    UnitType.MARINE: "mrn",
    UnitType.ZEALOT: "zlt",
    UnitType.STALKER: "stl",
    UnitType.ZERGLING: "zrg",
    UnitType.MEDIVAC: "mdv",
}
RENDER_FPS = 60


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
            for y in range(map_info.height):
                terrain_type = map_info.terrain[y][x]
                color = TERRAIN_COLORS[terrain_type]

                pygame.draw.rect(canvas, color,
                                 pygame.Rect((TILE_SIZE * x,
                                              TILE_SIZE * y),
                                             (TILE_SIZE, TILE_SIZE))),

        for unit in units:
            if unit.hp == 0:
                continue
            color = FACTION_COLORS[unit.faction]
            radius = TILE_SIZE * unit.radius
            center = np_to_pygame(unit.pos)
            pygame.draw.circle(canvas, color, center, radius)
            main_font_size = int(radius * 0.9)
            if unit.type not in self.fonts:
                text = UNIT_TYPE_ABBREVIATIONS[unit.type]
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
            cd_font = pygame.font.SysFont("Monospace",
                                          int(radius) // 2) \
                .render(f"{unit.cooldown:.02f}", True, (0, 0, 0),
                        (255, 255, 255, 230))
            canvas.blit(font, font.get_rect(center=center))
            canvas.blit(hp_font,
                        hp_font.get_rect(center=(center[0],
                                                 center[1] + main_font_size)))
            canvas.blit(cd_font,
                        cd_font.get_rect(center=(center[0],
                                                 center[1] - main_font_size)))
            if unit.target is not None:
                pygame.draw.line(canvas, (0, 0, 255),
                                 np_to_pygame(unit.pos),
                                 np_to_pygame(unit.target.pos))
            elif isinstance(unit.command, AttackMoveCommand):
                pygame.draw.line(canvas, (255, 0, 0),
                                 np_to_pygame(unit.pos),
                                 np_to_pygame(unit.command.pos))
            elif isinstance(unit.command, MoveCommand):
                pygame.draw.line(canvas, (0, 255, 0),
                                 np_to_pygame(unit.pos),
                                 np_to_pygame(unit.command.pos))

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


def np_to_pygame(np_vec):
    return tuple(TILE_SIZE * np_vec.astype(float))
