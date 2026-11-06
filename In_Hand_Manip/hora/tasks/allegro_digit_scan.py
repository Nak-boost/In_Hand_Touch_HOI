"""Dedicated scripted tabletop experiment: physical table/gel and inter-gel contacts."""
import copy
from .allegro_digit_tabletop import AllegroDigitTabletop
from hora.utils.hand_collision import collision_spec


class AllegroDigitScan(AllegroDigitTabletop):
    def _collision_spec(self, asset_path):
        spec = collision_spec(asset_path)
        spec.update(version='scan_full_table_contacts_v1', table_filter=0,
                    external_excluded_pairs=[], scene=copy.deepcopy(self.table_cfg))
        return spec
