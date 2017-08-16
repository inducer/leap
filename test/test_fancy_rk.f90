program test_rkmethod

  use RKMethod, only: dagrt_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_print_profile => print_profile, &
    timestep_shutdown => shutdown

  use sim_types

  implicit none

  type(region_type), target :: region
  type(region_type), pointer :: region_ptr

  type(sim_grid_state_type), pointer, dimension(:) :: initial_condition

  type(dagrt_state_type), target :: dagrt_state
  type(dagrt_state_type), pointer :: dagrt_state_ptr

  real*8 t_fin
  integer ntrips, igrid, idof
  parameter (t_fin=1d0, ntrips=20)

  integer istep

  ! start code ----------------------------------------------------------------

  dagrt_state_ptr => dagrt_state
  region_ptr => region
 
  region%n_grids = 2

  allocate(region%n_grid_dofs(region%n_grids))
  allocate(initial_condition(region%n_grids))

  do igrid = 1, region%n_grids
    region%n_grid_dofs(igrid) = 2
    allocate(initial_condition(igrid)%conserved_var(region%n_grid_dofs(igrid)))

    do idof = 1, region%n_grid_dofs(igrid)
      initial_condition(igrid)%conserved_var(idof) = 1
    end do
  end do

  call timestep_initialize( &
    region=region_ptr, &
    dagrt_state=dagrt_state_ptr, &
    state_y=initial_condition, &
    dagrt_t=0d0, &
    dagrt_dt=t_fin/20)

  do istep = 1,ntrips
    call timestep_run(region=region_ptr, dagrt_state=dagrt_state_ptr)
  enddo

  call timestep_print_profile(dagrt_state=dagrt_state_ptr)
  call timestep_shutdown(region=region_ptr, dagrt_state=dagrt_state_ptr)

end program

