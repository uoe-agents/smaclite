from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np

from smaclite.env.units.unit import Unit

INV_TIME_HORIZON = 1 / 2
USE_RVO = True


@dataclass
class Line:
    direction: np.ndarray = None
    point: np.ndarray = None


def det(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - a[1] * b[0]


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def velocity_computing_job(in_queue, out_queue):
    while True:
        idd, unit_ids, all_units = in_queue.get()
        if idd is None:
            break
        next_velocity = compute_new_velocity(idd, unit_ids, all_units)
        out_queue.put((idd, next_velocity))


def compute_new_velocity(idd: int, neighbour_ids: List[int],
                         all_units: List[Unit]):
    unit = all_units[idd]
    if unit.hp == 0:
        return unit.pref_velocity
    lines = []
    for idx in neighbour_ids:
        other = all_units[idx]
        if other == unit or other.hp == 0:
            continue
        relative_position = other.pos - unit.pos
        relative_velocity = unit.velocity - other.velocity
        distance_sq = np.inner(relative_position, relative_position)
        combined_radius = unit.radius + other.radius
        combined_radius_sq = combined_radius ** 2
        multiplier = 0.5

        line = Line()
        u = None

        if distance_sq > combined_radius_sq:
            # Vector from center of cutoff to relative velocity
            # (in velocity space)
            w = relative_velocity - INV_TIME_HORIZON * relative_position
            w_length_sq = np.inner(w, w)
            dot_product_1 = np.inner(w, relative_position)

            if dot_product_1 < 0 and \
                    dot_product_1 ** 2 > combined_radius_sq * w_length_sq:
                w_length = np.sqrt(w_length_sq)
                unit_w = w / w_length

                line.direction = np.array([unit_w[1], -unit_w[0]])
                u = (combined_radius * INV_TIME_HORIZON - w_length) * unit_w
            else:
                leg = np.sqrt(distance_sq - combined_radius_sq)
                if (det(relative_position, w) > 0):
                    # left leg
                    line.direction = np.array([
                        relative_position[0] * leg
                        - relative_position[1] * combined_radius,
                        relative_position[0] * combined_radius
                        + relative_position[1] * leg
                    ]) / distance_sq
                else:
                    line.direction = np.array([
                        - relative_position[0] * leg
                        - relative_position[1] * combined_radius,
                        relative_position[0] * combined_radius
                        - relative_position[1] * leg
                    ]) / distance_sq
                dot_product_2 = np.inner(relative_velocity, line.direction)

                # note: this only works because line.direction is a unit vector
                # so the dot product is the length of velocity's
                # shadow on the leg
                u = dot_product_2 * line.direction - relative_velocity
        else:
            inv_time_step = 16
            w = relative_velocity - inv_time_step * relative_position
            w_length = np.linalg.norm(w)
            unit_w = w / w_length

            line.direction = np.array([unit_w[1], -unit_w[0]])
            u = (combined_radius * inv_time_step - w_length) * unit_w
        line.point = unit.velocity + multiplier * u
        lines.append(line)

    lines_successful, result = linear_program_2(lines, unit.max_velocity,
                                                unit.pref_velocity, False)
    if lines_successful < len(lines):
        result = linear_program_3(lines, lines_successful,
                                  unit.max_velocity, result)

    return result


def linear_program_1(lines: List[Line], line_no: int, max_velocity: float,
                     pref_velocity: np.ndarray,
                     optimize_direction: bool) -> Tuple[bool, np.ndarray]:
    line = lines[line_no]
    dot_product = np.inner(line.point, line.direction)
    discriminant = dot_product ** 2 + max_velocity ** 2 \
        - np.inner(line.point, line.point)

    if discriminant < 0:
        return False, None

    sqrt_discriminant = np.sqrt(discriminant)
    t_left = -dot_product - sqrt_discriminant
    t_right = -dot_product + sqrt_discriminant

    for other in lines[:line_no]:
        denominator = det(line.direction, other.direction)
        numerator = det(other.direction, line.point - other.point)

        if abs(denominator) < 1e-6:
            # lines are parallel
            if numerator < 0:
                return False, None
            continue
        t = numerator / denominator
        if denominator > 0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)

        if t_left > t_right:
            return False, None

    if optimize_direction:
        return (True, line.point + t_right * line.direction) \
            if np.inner(pref_velocity, line.direction) > 0 \
            else (True, line.point + t_left * line.direction)

    t = np.inner(line.direction,  pref_velocity - line.point)
    if t < t_left:
        return True, line.point + t_left * line.direction
    elif t > t_right:
        return True, line.point + t_right * line.direction
    return True, line.point + t * line.direction


def linear_program_2(lines: List[Line], max_velocity: float,
                     pref_velocity: np.ndarray,
                     optimize_direction: bool,
                     result: float = 0.0) -> Tuple[int, np.ndarray]:
    if optimize_direction:
        result = pref_velocity * max_velocity
    elif np.inner(pref_velocity, pref_velocity) > max_velocity**2:
        result: np.ndarray = normalize(pref_velocity) * max_velocity
    else:
        result = pref_velocity

    for i, line in enumerate(lines):
        if det(line.direction, line.point - result) <= 0:
            # already outside
            continue
        lp1_ok, lp1_result = linear_program_1(lines, i, max_velocity,
                                              pref_velocity,
                                              optimize_direction)
        if not lp1_ok:
            return i, result
        result = lp1_result
    return len(lines), result


def linear_program_3(lines: List[Line], lines_successful: int,
                     max_velocity: float,
                     current_result: np.ndarray) -> np.ndarray:
    distance = 0.0
    result = current_result

    for i in range(lines_successful, len(lines)):
        line = lines[i]
        if det(line.direction, line.point - line.point) <= distance:
            # already outside
            continue
        new_lines = []
        for other in lines[:i]:
            new_line = Line()
            determinant = det(line.direction, other.direction)
            if abs(determinant) < 1e-6:
                # lines are parallel
                if np.inner(line.direction, other.direction) > 0:
                    # lines are in the same direction
                    continue
                else:
                    new_line.point = 0.5 * (line.point + other.point)
            else:
                new_line.point = line.point + det(other.direction,
                                                  line.point - other.point) \
                    / determinant * line.direction
            new_line.direction = normalize(other.direction - line.direction)
            new_lines.append(new_line)
        tmp_result = current_result.copy()
        lines_successful, result = linear_program_2(new_lines, max_velocity,
                                                    np.array([-line.direction[1],
                                                              line.direction[0]],),
                                                    True, current_result)
        if lines_successful < len(new_lines):
            result = tmp_result

        distance = det(line.direction, line.point - result)
    return result
