!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!            SHELL ROUTINES IN FORTRAN  TO SPEED UP THE RK INTEGRATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!            POUR COMPILER : f2py -c -m Shell4KH_boost Shell4KH_boost.f90 --fcompiler=gnu95 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!LES TABLEAUX DE COMPLEXES SONT REPRESENTES PAR DES REELS
! POUR DES RAISONS MYSTERIEUSES, F2PY SEMBLE BUGGER SINON 

!!RANDOM NUMBERS
SUBROUTINE init_random_seed()
	INTEGER :: i, n, clock
	INTEGER, DIMENSION(:), ALLOCATABLE :: seed
	CALL RANDOM_SEED(size = n)
	ALLOCATE(seed(n))

	CALL SYSTEM_CLOCK(COUNT=clock)
	seed = clock + 37 * (/ (i - 1, i = 1, n) /)
	CALL RANDOM_SEED(PUT = seed)

	DEALLOCATE(seed)
END SUBROUTINE

!On génere un entier entre 0 et NMAX-1
subroutine Randn(zu,zv)
implicit none 
	real(8), intent(out)::zu,zv
	real(8)::u,v
        real(8):: PI_8  = 4 * atan (1.0_8)
	call random_number(u)
	call random_number(v)
        zu=-2.*log(u)*cos(2*pi_8*v)
        zv=-2.*log(u)*sin(2*pi_8*v)
end subroutine


!! RHS NON-LINEAIRE
SUBROUTINE RHS(da,a,gam,lam,Etas,n)
    implicit none
    integer,intent(in):: n
    real(8),intent(in):: gam,lam
    real(8),intent(in),dimension(0:n-1):: Etas
    real(8),intent(in),dimension(0:n-1,0:1):: a
    real(8),intent(out),dimension(0:n-1,0:1):: da
    real(8):: coeff
    integer:: k,km,kp

        do k =0,(n-1)
                da(k,0)=0.
                da(k,1)=0.
                if (k<n-2) then  !a(k+1)*a(k+2)
                        km=k+1
                        kp=k+2
                        coeff=(lam**2)*gam
                        da(k,0) =da(k,0)+ coeff * (a(km,0)*a(kp,0)-a(km,1)*a(kp,1))
                        da(k,1) =da(k,1)+ coeff * (-a(km,0)*a(kp,1)-a(km,1)*a(kp,0))
                endif

                if ((k<n-1) .AND. (k>0)) then !a(k-1)*a(k+1)
                        km=k-1
                        kp=k+1
                        coeff=(-lam)*(1.+gam)
                        da(k,0) =da(k,0)+ coeff * (a(km,0)*a(kp,0)-a(km,1)*a(kp,1))
                        da(k,1) =da(k,1)+ coeff * (-a(km,0)*a(kp,1)-a(km,1)*a(kp,0))
                endif

                if (k >1) then !a(k-2)a(k-1)
                        km=k-2
                        kp=k-1
                        coeff=1.
                        da(k,0) =da(k,0)+ coeff * (a(km,0)*a(kp,0)-a(km,1)*a(kp,1))
                        da(k,1) =da(k,1)+ coeff * (-a(km,0)*a(kp,1)-a(km,1)*a(kp,0))
                endif

                da(k,0)=da(k,0)*Etas(k) !+ f(k,0) !+ dissip(k)*da(k,0)
                da(k,1)=da(k,1)*Etas(k) !+ f(k,1) !+ dissip(k)*da(k,1)
        enddo
END SUBROUTINE
 
 
!! UPDATE VIA RUNGE KUTTA
SUBROUTINE UPDATE(da,a,dt,nt,f,dissip,gam,lam,Etas,n)
        implicit none
        integer,intent(in):: n,nt
        real(8),intent(in):: dt,gam,lam
        real(8),intent(in),dimension(0:n-1):: dissip,Etas
        real(8),intent(in),dimension(0:n-1):: f
        real(8),intent(inout),dimension(0:n-1,0:1):: a,da
        !f2py intent(in,out,inplace) :: a,da
        real(8),dimension(0:n-1,0:1):: a1,a2,a3,a4
	real(8)::zu,zv

        integer:: i,k
    
        do i=1,nt

            !transfer nl
            call rhs(a1,a,gam,lam,etas,n)
            call rhs(a2,a+0.5*dt*a1,gam,lam,etas,n)
            call rhs(a3,a+0.5*dt*a2,gam,lam,etas,n)
            call rhs(a4,a+ dt*a3,gam,lam,etas,n)
            da=(a1+2.*a2+ 2.*a3+a4)/6.
            !dissip
            a(:,0)=(a(:,0)+dt*da(:,0))*exp(-dissip(:)*dt)
            a(:,1)=(a(:,1)+dt*da(:,1))*exp(-dissip(:)*dt)
            da=a1

            !forcing
            do k=0,n-1
                if (f(k)>0.) then
                	call randn(zu,zv)
                        a(k,0)=a(k,0)+sqrt(2*f(k)*dt)*zu
                        a(k,1)=a(k,1)+sqrt(2*f(k)*dt)*zv
                endif
            enddo

        enddo

END SUBROUTINE

