

from smaclite.env.maps.map import MapInfo, TerrainType
import pygame

TILE_SIZE = 32
TERRAIN_COLORS = {
    TerrainType.NORMAL: (0, 255, 0),
    TerrainType.CLIFF: (255, 0, 0),
    TerrainType.NONE: (0, 0, 0),
}
RENDER_FPS = 60


class Renderer:
    def render(self, map_info: MapInfo):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = self.__create_window(map_info)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(TILE_SIZE * (map_info.width, map_info.height))
        for x in range(map_info.width):
            for y in range(map_info.height):
                terrain_type = map_info.terrain[x][y]
                color = TERRAIN_COLORS[terrain_type]

                pygame.draw.rect(canvas, color,
                                 pygame.Rect(TILE_SIZE * (x, y),
                                             (TILE_SIZE, TILE_SIZE))),

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(RENDER_FPS)

    def close(self):
        if self.windows is not None:
            pygame.display.quit()
            pygame.quit()

    def __create_window(self, map_info: MapInfo):
        return pygame.display.set_mode(TILE_SIZE * (map_info.width,
                                                    map_info.height))
