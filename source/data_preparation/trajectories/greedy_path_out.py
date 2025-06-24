def find_exit_path(
    density_grid,
    threshold=0.2,
    max_steps=1000,
    grid_spacing=0.5,
    max_radius_angs=3.0,
    escape_patience=50,
    allowed_backward_angs=0.0,
    start=None,
    target=torch.tensor([-10.5, 4.5, -11.5]),
    direction_weight=0.2,  # how strongly to bias toward the target
    min_density=0.05
):
    import torch
    import numpy as np
    import random

    origin = torch.tensor([-20.0, -20.0, -20.0])
    total_density = 0
    grid_shape = density_grid.shape
    visited = set()

    if start is None:
        max_r = 6.0
        coords = []
        for x in np.arange(-max_r, max_r + grid_spacing, grid_spacing):
            for y in np.arange(-max_r, max_r + grid_spacing, grid_spacing):
                for z in np.arange(-max_r, max_r + grid_spacing, grid_spacing):
                    if np.linalg.norm([x, y, z]) <= max_r:
                        coords.append([x, y, z])
        start_coord = torch.tensor(random.choice(coords), dtype=torch.float32)
        print('start_coord', start_coord, 'with len', torch.norm(start_coord))
    else:
        start_coord = torch.tensor(start, dtype=torch.float32)
        print('start_coord', start_coord, 'with len', torch.norm(start_coord))

    path = [start_coord.tolist()]
    current = start_coord.clone()
    visited.add(tuple(((current - origin) / grid_spacing).round().tolist()))

    no_progress_steps = 0
    voxel_radius = int(max_radius_angs / grid_spacing)
    start_tensor = start_coord.clone()

    first_step = 0
    for step in range(max_steps):
        current_idx = ((current - origin) / grid_spacing).round().to(torch.long)
        current_density = density_grid[tuple(current_idx.tolist())]
        current_dist = torch.norm(current - start_tensor)

        best_score = -float("inf")
        best_neighbor = None

        direction_to_target = (target - current)
        direction_to_target = direction_to_target / (torch.norm(direction_to_target) + 1e-8)

        for dx in range(-voxel_radius, voxel_radius + 1):
            for dy in range(-voxel_radius, voxel_radius + 1):
                for dz in range(-voxel_radius, voxel_radius + 1):
                    if dx == dy == dz == 0:
                        continue

                    offset = torch.tensor([dx, dy, dz], dtype=torch.float32) * grid_spacing
                    neighbor = current + offset
                    neighbor_idx = ((neighbor - origin) / grid_spacing).round().to(torch.long)

                    if not all(0 <= i < s for i, s in zip(neighbor_idx, grid_shape)):
                        continue

                    neighbor_key = tuple(neighbor_idx.tolist())
                    if neighbor_key in visited:
                        continue

                    density = density_grid[neighbor_key]
                    dist_from_start = torch.norm(neighbor - start_tensor)

                    if dist_from_start < current_dist - allowed_backward_angs:
                        continue

                    if density < 0.75 * current_density:
                        continue

                    if density < min_density and first_step==0:
                        continue

                    # if density > 0.5 * current_density:
                    #     continue

                    # if density > 0.15:
                    #     continue

                    move_dir = (neighbor - current)
                    move_dir = move_dir / (torch.norm(move_dir) + 1e-8)
                    dir_alignment = torch.dot(move_dir, direction_to_target)

                    # TODO I forget why I used -density and not +density
                    score = -density + direction_weight * dir_alignment
                    print('score', score, density, best_score)

                    if score > best_score:
                        best_score = score
                        best_neighbor = (neighbor, neighbor_key, dist_from_start, density)

        if best_neighbor:
            neighbor, idx_tuple, dist, dens = best_neighbor
            visited.add(idx_tuple)
            path.append(neighbor.tolist())
            total_density += dens
            print(f"Step {step}: moved to {neighbor.tolist()} with density {dens:.3f}, dist from origin {torch.norm(neighbor):.3f}")
            current = neighbor
            no_progress_steps = 0
            first_step = 1
        else:
            no_progress_steps += 1
            if no_progress_steps >= escape_patience:
                print("Gave up: no better neighbor after", escape_patience, "steps.")
                break
            else:
                continue

        if torch.norm(current - torch.tensor([0, 0, 0])) > 18:
            print("Escaped bounding box.", current)
            break

    if torch.norm(current - torch.tensor([0, 0, 0])) < 10:
        total_density = 0
    print('total_density', total_density)
    return path, total_density

paths = []
total_densitys = []
best_total_density = 0
# for min_density in [1, 0.75, 0.5, 0.25, 0.1, 0.075, 0.05, .025, 0.01]:
for min_density in [0.25, 0.1, 0.075, 0.5, 0.025]:
    for i in range(0,5):
        path, total_density = find_exit_path(density, threshold=0.0001, max_steps=10000, grid_spacing=0.5, allowed_backward_angs=0.1, min_density = min_density, target=torch.tensor([-1.67086515, -2.68134037, -24.79956902]), direction_weight=0.5, start=[0,0,0]) #, start=[-5., 0.5, -1]
        total_densitys.append(total_density)
        paths.append(path)
        if total_density > best_total_density:
            best_total_density = total_density
            print("\t\t best_total_density:", best_total_density)
