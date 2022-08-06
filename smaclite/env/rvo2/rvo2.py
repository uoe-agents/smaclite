from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from smaclite.env.rvo2.static_obstacle import StaticObstacle
from smaclite.env.units.unit import Unit

TAU = 1
INV_TIME_HORIZON = 1 / TAU
TAU_OBST = 1
INV_TIME_HORIZON_OBST = 1 / TAU_OBST


@dataclass
class Line:
    direction: np.ndarray = None
    point: np.ndarray = None


def det(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - a[1] * b[0]


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def vec_len_sq(v: np.ndarray) -> float:
    return np.inner(v, v)


def left_of(a, b, c):
    return det(a - c, b - a) < 0


def compute_new_velocity(unit: Unit, neighbours: List[Tuple[Unit, float]],
                         obstacle_neighbours: List[StaticObstacle]):
    """This is, for the most part, a direct port of the C++ RVO2 library.
    All credits for the original code go to:
    https://gamma.cs.unc.edu/RVO2/

    Computes an adjusted velocity that takes into account collision avoidance,
    using ORCA (Optimal Reciprocal Collision Avoidance).

    Args:
        idd (int): the index of the unit for which the velocity is computed
        neighbour_ids (List[int]): the indices of the neighbouring units
        all_units (List[Unit]): the list of all units

    Returns:
        np.ndarray: the new velocity for the unit
    """
    if not unit.pref_velocity.any():
        return unit.pref_velocity
    if unit.hp == 0:
        return unit.pref_velocity
    obstacle_lines = []
    unit_lines = []

    # Static obstacle lines
    for o in obstacle_neighbours:
        for i, o1 in enumerate(o.lines):
            o2 = o.lines[(i + 1) % len(o.lines)]
            if not left_of(o1.point, o2.point, unit.pos):
                continue
            rel_pos_1 = o1.point - unit.pos
            rel_pos_2 = o2.point - unit.pos

            if any(det(INV_TIME_HORIZON_OBST * rel_pos_1
                       - line.point,
                       line.direction)
                   - INV_TIME_HORIZON_OBST * unit.radius >= -1e-5
                   and det(INV_TIME_HORIZON_OBST * rel_pos_2
                           - line.point,
                           line.direction)
                   - INV_TIME_HORIZON_OBST * unit.radius >= -1e-5
                   for line in obstacle_lines):
                # Already covered by a previous obstacle
                continue

            dist_sq_1 = vec_len_sq(rel_pos_1)
            dist_sq_2 = vec_len_sq(rel_pos_2)

            radius_sq = unit.radius_sq

            obstacle_vector = o2.point - o1.point
            s = np.inner(-rel_pos_1, obstacle_vector) \
                / vec_len_sq(obstacle_vector)
            dist_sq_line = vec_len_sq(-rel_pos_1
                                      - np.inner(s, obstacle_vector))

            line = Line()

            if s < 0 and dist_sq_1 < radius_sq:
                line.point = np.zeros(2)
                line.direction = normalize(np.array([-rel_pos_1[1],
                                                     rel_pos_1[0]]))
                obstacle_lines.append(line)
                continue
            elif s > 1 and dist_sq_2 <= radius_sq:
                if det(rel_pos_2, o2.unit_direction) < 0:
                    continue
                line.point = np.zeros(2)
                line.direction = normalize(np.array([-rel_pos_2[1],
                                                     rel_pos_2[0]]))
                obstacle_lines.append(line)
                continue
            elif s > 0 and s < 1 and dist_sq_line <= radius_sq:
                line.point = np.zeros(2)
                line.direction = -o1.unit_direction
                obstacle_lines.append(line)
                continue

            left_leg_dir = right_leg_dir = None
            if s < 0 and dist_sq_line <= radius_sq:
                o2 = o1
                leg_1 = np.sqrt(dist_sq_1 - radius_sq)
                left_leg_dir = np.array([rel_pos_1[0] * leg_1
                                         - rel_pos_1[1] * unit.radius,
                                         rel_pos_1[0] * unit.radius
                                         + rel_pos_1[1] * leg_1]) / dist_sq_1
                right_leg_dir = np.array([rel_pos_1[0] * leg_1
                                          + rel_pos_1[1] * unit.radius,
                                          -rel_pos_1[0] * unit.radius
                                          + rel_pos_1[1] * leg_1]) / dist_sq_1
            elif s > 1 and dist_sq_line <= radius_sq:
                o1 = o2
                leg_2 = np.sqrt(dist_sq_2 - radius_sq)
                left_leg_dir = np.array([rel_pos_2[0] * leg_2
                                         - rel_pos_2[1] * unit.radius,
                                         rel_pos_2[0] * unit.radius
                                         + rel_pos_2[1] * leg_2]) / dist_sq_2
                right_leg_dir = np.array([rel_pos_2[0] * leg_2
                                          + rel_pos_2[1] * unit.radius,
                                          -rel_pos_2[0] * unit.radius
                                          + rel_pos_2[1] * leg_2]) / dist_sq_2
            else:
                leg_1 = np.sqrt(dist_sq_1 - radius_sq)
                left_leg_dir = np.array([rel_pos_1[0] * leg_1
                                         - rel_pos_1[1] * unit.radius,
                                         rel_pos_1[0] * unit.radius
                                         + rel_pos_1[1] * leg_1]) / dist_sq_1
                leg_2 = np.sqrt(dist_sq_2 - radius_sq)
                right_leg_dir = np.array([rel_pos_2[0] * leg_2
                                          + rel_pos_2[1] * unit.radius,
                                          -rel_pos_2[0] * unit.radius
                                          + rel_pos_2[1] * leg_2]) / dist_sq_2

            left_neighbour = o.lines[i - 1]
            is_left_leg_foreign = False
            is_right_leg_foreign = False

            if det(left_leg_dir, -left_neighbour.unit_direction) >= 0:
                left_leg_dir = -left_neighbour.unit_direction
                is_left_leg_foreign = True
            if det(right_leg_dir, o2.unit_direction) <= 0:
                right_leg_dir = o2.unit_direction
                is_right_leg_foreign = True

            left_cutoff = INV_TIME_HORIZON_OBST * (o1.point - unit.pos)
            right_cutoff = INV_TIME_HORIZON_OBST * (o2.point - unit.pos)
            cutoff_vector = right_cutoff - left_cutoff

            t = 0.5 if o1 is o2 else (np.inner(unit.velocity - left_cutoff,
                                               cutoff_vector) /
                                      vec_len_sq(cutoff_vector))
            t_left = np.inner(unit.velocity - left_cutoff,
                              left_leg_dir)
            t_right = np.inner(unit.velocity - right_cutoff,
                               right_leg_dir)

            if (t < 0 and t_left < 0) \
                    or (o1 is o2 and t_left < 0 and t_right < 0):
                unit_w = normalize(unit.velocity - left_cutoff)
                line.direction = np.array([unit_w[1], -unit_w[0]])
                line.point = left_cutoff + \
                    unit.radius * INV_TIME_HORIZON_OBST * unit_w
                obstacle_lines.append(line)
                continue
            elif t > 1 and t_right < 0:
                unit_w = normalize(unit.velocity - right_cutoff)
                line.direction = np.array([unit_w[1], -unit_w[0]])
                line.point = right_cutoff + \
                    unit.radius * INV_TIME_HORIZON_OBST * unit_w
                obstacle_lines.append(line)
                continue

            dist_sq_cutoff = np.inf if t < 0 or t > 1 or o1 is o2 \
                else vec_len_sq(unit.velocity - (left_cutoff +
                                                 t * cutoff_vector))
            dist_sq_left = np.inf if t_left < 0 \
                else vec_len_sq(unit.velocity - (left_cutoff
                                                 + t_left * left_leg_dir))
            dist_sq_right = np.inf if t_right < 0 \
                else vec_len_sq(unit.velocity - (right_cutoff
                                                 + t_right * right_leg_dir))

            if dist_sq_cutoff <= dist_sq_left \
                    and dist_sq_cutoff <= dist_sq_right:
                line.direction = -o1.unit_direction
                line.point = left_cutoff + \
                    unit.radius * INV_TIME_HORIZON_OBST * \
                    np.array(-line.direction[1], line.direction[0])
                obstacle_lines.append(line)
                continue
            elif dist_sq_left <= dist_sq_right:
                if is_left_leg_foreign:
                    continue
                line.direction = left_leg_dir
                line.point = left_cutoff + \
                    unit.radius * INV_TIME_HORIZON_OBST * \
                    np.array(-line.direction[1], line.direction[0])
                obstacle_lines.append(line)
                continue
            else:
                if is_right_leg_foreign:
                    continue
                line.direction = -right_leg_dir
                line.point = right_cutoff + \
                    unit.radius * INV_TIME_HORIZON_OBST * \
                    np.array(-line.direction[1], line.direction[0])
                obstacle_lines.append(line)
                continue

    # Unit lines
    for other, distance in neighbours:
        if other is unit or other.hp == 0:
            continue
        relative_position = other.pos - unit.pos
        relative_velocity = unit.velocity - other.velocity
        distance_sq = distance ** 2
        combined_radius = unit.radius + other.radius
        combined_radius_sq = combined_radius ** 2
        # If the other unit is moving too, assume
        # it will move out of the way by half.
        # Otherwise, assume it will stand still, and move out of the way fully.
        multiplier = 0.5 if other.pref_velocity.any() else 1
        line_list = unit_lines

        line = Line()
        u = None

        if distance_sq > combined_radius_sq:
            # Vector from center of cutoff to relative velocity
            # (in velocity space)
            w = relative_velocity - INV_TIME_HORIZON * relative_position
            w_length_sq = vec_len_sq(w)
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
            inv_time_step = 16  # Number of ticks per second
            w = relative_velocity - inv_time_step * relative_position
            w_length = np.linalg.norm(w)
            unit_w = w / w_length

            line.direction = np.array([unit_w[1], -unit_w[0]])
            u = (combined_radius * inv_time_step - w_length) * unit_w
        line.point = unit.velocity + multiplier * u
        line_list.append(line)
    num_obstacle_lines = len(obstacle_lines)
    lines = obstacle_lines + unit_lines

    lines_successful, result = linear_program_2(lines, unit.max_velocity,
                                                unit.pref_velocity, False)
    if lines_successful < len(lines):
        result = linear_program_3(lines, num_obstacle_lines,
                                  lines_successful, unit.max_velocity,
                                  result)

    return result


def linear_program_1(lines: List[Line], line_no: int, max_velocity: float,
                     pref_velocity: np.ndarray,
                     optimize_direction: bool) -> Tuple[bool, np.ndarray]:
    line = lines[line_no]
    dot_product = np.inner(line.point, line.direction)
    discriminant = dot_product ** 2 + max_velocity ** 2 \
        - vec_len_sq(line.point)

    if discriminant < 0:
        return False, None

    sqrt_discriminant = np.sqrt(discriminant)
    t_left = -dot_product - sqrt_discriminant
    t_right = -dot_product + sqrt_discriminant

    for other in lines[:line_no]:
        denominator = det(line.direction, other.direction)
        numerator = det(other.direction, line.point - other.point)

        if abs(denominator) < 1e-5:
            # lines are parallel
            if numerator < 0:
                return False, None
            continue
        t = numerator / denominator
        if denominator >= 0:
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
    elif vec_len_sq(pref_velocity) > max_velocity**2:
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


def linear_program_3(lines: List[Line], num_obstacle_lines: int,
                     num_success: int, max_velocity: float,
                     current_result: np.ndarray) -> np.ndarray:
    distance = 0.0
    result = current_result

    for i in range(num_success, len(lines)):
        line = lines[i]
        if det(line.direction, line.point - result) <= distance:
            # already outside
            continue
        new_lines = lines[:num_obstacle_lines]
        for other in lines[num_obstacle_lines:i]:
            new_line = Line()
            determinant = det(line.direction, other.direction)
            if abs(determinant) <= 1e-5:
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
        tmp_result = result.copy()
        num_success, result = linear_program_2(new_lines, max_velocity,
                                               np.array([-line.direction[1],
                                                         line.direction[0]]),
                                               True, result)
        if num_success < len(new_lines):
            result = tmp_result

        distance = det(line.direction, line.point - result)
    return result
