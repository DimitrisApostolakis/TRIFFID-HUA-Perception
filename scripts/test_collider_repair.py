import unittest
import numpy as np
from scipy.spatial import cKDTree
from collider_repair import repair_small_holes, topology, _triangulate


def sheet():
    vertices = np.array([[x,y,0.] for y in range(5) for x in range(5)])
    triangles = []
    for y in range(4):
        for x in range(4):
            if (x,y) == (1,1): continue
            a=y*5+x; triangles.extend([[a,a+1,a+6],[a,a+6,a+5]])
    return vertices,np.array(triangles)


class RepairTests(unittest.TestCase):
    def repair(self,v,t,**kwargs):
        x,y=np.meshgrid(np.linspace(0,4,81),np.linspace(0,4,81))
        tree=cKDTree(np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)]))
        defaults=dict(max_diameter=2,max_planarity_error=.02,max_support_distance=.1)
        defaults.update(kwargs)
        return repair_small_holes(v,t,tree,**defaults)

    def test_closes_hole_without_moving_existing_geometry(self):
        v,t=sheet();v0=v.copy();out,r=self.repair(v,t)
        np.testing.assert_array_equal(v,v0);np.testing.assert_array_equal(out[:len(t)],t)
        self.assertEqual(r['holes_filled'],1);self.assertEqual(r['added_triangles'],2)
        self.assertEqual(r['before']['boundary_edges']-r['after']['boundary_edges'],4)
        self.assertTrue(np.all(np.cross(v[out[:,1]]-v[out[:,0]],v[out[:,2]]-v[out[:,0]])[:,2]>0))

    def test_outside_border_not_capped(self):
        v,t=sheet();out,r=self.repair(v,t,max_diameter=10)
        self.assertEqual(r['holes_filled'],1)
        self.assertTrue(set(r['rejected']) & {'outside_border_or_curved_surface','duplicate_faces'})
        self.assertEqual(topology(out)['boundary_edges'],16)

    def test_size_budget_and_support_limits(self):
        v,t=sheet()
        for kwargs in [dict(max_diameter=.5),dict(max_added_triangles=1),dict(max_support_distance=.0001)]:
            out,r=self.repair(v,t,**kwargs);self.assertEqual(r['holes_filled'],0)

    def test_nonplanar_hole_not_filled(self):
        v,t=sheet();v[6,2]=.2
        out,r=self.repair(v,t);self.assertEqual(r['holes_filled'],0)

    def test_self_intersection_rejected(self):
        self.assertIsNone(_triangulate(np.array([[0.,0],[1,1],[0,1],[1,0]])))

    def test_concave_polygon_and_reverse_winding(self):
        p=np.array([[0.,0],[2,0],[2,2],[1,1],[0,2]])
        for polygon in [p,p[::-1]]:
            t=_triangulate(polygon);self.assertEqual(len(t),3)
            area=sum(abs(np.linalg.det(np.stack([polygon[b]-polygon[a],polygon[c]-polygon[a]])))*.5 for a,b,c in t)
            self.assertAlmostEqual(area,3.)

    def test_invalid_limits_rejected(self):
        v,t=sheet()
        with self.assertRaises(ValueError):self.repair(v,t,max_diameter=float('nan'))

if __name__ == '__main__':unittest.main()
